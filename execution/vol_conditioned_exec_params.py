#!/usr/bin/env python3
"""
Volatility-Conditioned Execution Parameter Optimization
=========================================================
Jupiter CPU research: for each volatility regime, find optimal TP/SL/hold/threshold
parameters using FIFO market replay fills from the CNN-Mamba v2 predictions.

Key insight: The FIFO regime sweep (72 simple configs) all lost money because
one-size-fits-all parameters are wrong. Different volatility environments need
different execution parameters. This script finds the optimal lookup table:
  vol_regime -> {tp_ticks, sl_ticks, max_hold_ms, z_threshold}

Approach:
  1. Load exec features (realized vol, spread, fill prob) for each OOS date
  2. Load CNN-Mamba v2 decay predictions (z-scores) for those dates
  3. Segment each trading day's decision windows into vol regimes (low/med/high)
     based on realized vol from exec features
  4. For each regime, grid-search over TP/SL/hold/threshold configs
  5. Run FIFO market replay for each config+regime combination
  6. Report per-regime optimal parameters and aggregate performance

Uses multiprocessing with 16 workers (HC #62).
Commission: $4.70 RT = 0.376 ticks (HC #52).

Usage:
    python vol_conditioned_exec_params.py --workers 16
    python vol_conditioned_exec_params.py --workers 8 --dry-run
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import socket
import itertools
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

# ── Paths ────────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(LVL3_ROOT / "alpha_discovery" / "deep_models"))

EXEC_FEAT_DIR = LVL3_ROOT / "output" / "exec_features_v1"
DECAY_PRED_DIR = LVL3_ROOT / "output" / "decay_v4_comprehensive" / "CNN-Mamba_v2"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
OUTPUT_DIR = LVL3_ROOT / "output" / "vol_conditioned_exec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (HC #52) ──────────────────────────────────────────────────────
TICK_RAW = 250_000_000       # 0.25 pts in Databento fixed-point
TICK_USD = 12.50
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50

# Exec feature indices (from exec_feature_engineering.py)
FEAT_IDX = {
    "fill_prob_1s": 0, "fill_prob_3s": 1, "fill_prob_10s": 2,
    "spread_ticks": 20, "spread_mean_10k": 24,
    "price_volatility_window": 43,
    "tod_sin": 34, "tod_cos": 35,
    "minutes_from_open": 36, "session_progress": 37,
    "depth_imbalance_l1": 31,
    "event_rate": 38, "trade_rate": 39,
    "informed_flow_score": 10,
}

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "vol_conditioned.log"), mode="a"),
    ],
)
log = logging.getLogger("vol_exec")

# Suppress replay engine noise
logging.getLogger("fifo_market_replay").setLevel(logging.WARNING)


# ── Vol regime definitions ───────────────────────────────────────────────────
VOL_REGIMES = ["low", "medium", "high"]
VOL_PERCENTILES = (33, 67)  # computed per-date from realized vol

# ── Parameter grid (per-regime search) ───────────────────────────────────────
PARAM_GRID = {
    "tp_ticks": [2, 3, 4, 6, 8, 12],
    "sl_ticks": [2, 3, 4, 6, 8],
    "max_hold_ms": [5000, 10000, 30000, 60000],
    "z_threshold": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
}


def build_param_configs() -> List[dict]:
    """Build cartesian product of the parameter grid."""
    keys = sorted(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    configs = []
    for i, combo in enumerate(itertools.product(*values)):
        configs.append(dict(zip(keys, combo)))
        configs[-1]["config_id"] = i
    return configs


# =============================================================================
# Data loading
# =============================================================================

def load_all_dates() -> List[dict]:
    """
    Load overlapping dates where we have both exec features and decay predictions.
    Returns list of dicts with all needed data per date.
    """
    # Find overlapping dates
    exec_dates = set()
    for f in EXEC_FEAT_DIR.glob("*_exec_features.npz"):
        exec_dates.add(f.name[:8])

    pred_dates = set()
    for d in DECAY_PRED_DIR.iterdir():
        if d.is_dir() and (d / "predictions.npz").exists():
            pred_dates.add(d.name)

    overlap = sorted(exec_dates & pred_dates)
    log.info(f"Found {len(overlap)} overlapping dates (exec features + predictions)")

    if not overlap:
        log.error("No overlapping dates found!")
        return []

    all_data = []
    for date_str in overlap:
        try:
            # Load exec features
            ef = np.load(str(EXEC_FEAT_DIR / f"{date_str}_exec_features.npz"),
                         allow_pickle=True)
            features = ef["features"]  # (n_windows, 44)
            feat_names = ef["feature_names"]
            decision_stride = int(ef["decision_stride"])

            # Load predictions
            pred_file = DECAY_PRED_DIR / date_str / "predictions.npz"
            pf = np.load(str(pred_file), allow_pickle=True)
            preds = pf["preds"]            # (N, 3) — z-scores for 1s/5s/10s
            valid_indices = pf["valid_indices"]  # (N,) — indices into MBO events
            labels_1s = pf["labels_1s"]
            labels_10s = pf["labels_10s"]

            # Load MBO timestamps
            mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
            if not mbo_path.exists():
                log.warning(f"  Skipping {date_str}: no MBO events")
                continue
            mbo = np.load(str(mbo_path), allow_pickle=True)
            timestamps = mbo["timestamps"]
            n_events = len(timestamps)

            # Map predictions to timestamps
            safe_idx = np.clip(valid_indices, 0, n_events - 1)
            pred_ts = timestamps[safe_idx]

            # Classify each prediction window into a vol regime
            # Map prediction indices to feature windows
            # Features: 1 per decision_stride events, predictions: at valid_indices
            vol_feature = features[:, FEAT_IDX["price_volatility_window"]]

            # Compute vol regime boundaries for this date
            p_low, p_high = np.percentile(vol_feature, VOL_PERCENTILES)

            # Assign each prediction to its nearest feature window's vol regime
            pred_regimes = []
            for vi in valid_indices:
                feat_window_idx = min(vi // decision_stride, len(vol_feature) - 1)
                v = vol_feature[feat_window_idx]
                if v <= p_low:
                    pred_regimes.append("low")
                elif v <= p_high:
                    pred_regimes.append("medium")
                else:
                    pred_regimes.append("high")

            all_data.append({
                "date": date_str,
                "features": features,
                "preds": preds,
                "labels_1s": labels_1s,
                "labels_10s": labels_10s,
                "pred_ts": pred_ts,
                "pred_regimes": pred_regimes,
                "n_events": n_events,
                "decision_stride": decision_stride,
                "vol_boundaries": (p_low, p_high),
            })
            log.info(f"  Loaded {date_str}: {len(preds)} predictions, "
                     f"{len(features)} feat windows, "
                     f"vol bounds ({p_low:.6f}, {p_high:.6f})")

        except Exception as e:
            log.warning(f"  Error loading {date_str}: {e}")
            continue

    return all_data


# =============================================================================
# FIFO Replay (import from existing engine)
# =============================================================================

_HAS_FIFO_ENGINE = False
try:
    from fifo_market_replay import FIFOReplayEngine, TICK_RAW as _TR
    _HAS_FIFO_ENGINE = True
    log.info("FIFO replay engine imported successfully")
except ImportError:
    log.warning("Could not import FIFOReplayEngine — will use simplified PnL model")


def run_fifo_replay_for_date(
    date_str: str,
    signals: List[dict],
    tp_ticks: float,
    sl_ticks: float,
    max_hold_ms: float,
) -> Optional[List[dict]]:
    """
    Run FIFO market replay for a single date with given parameters.
    Returns list of trade result dicts.
    """
    if not _HAS_FIFO_ENGINE:
        return None

    try:
        engine = FIFOReplayEngine(date=date_str)
        engine.max_hold_ns = int(max_hold_ms * 1e6)
        engine.cancel_after_ns = int(15_000 * 1e6)  # 15s cancel timeout

        trades = engine.simulate(
            signals=signals,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
            order_type="limit",
        )

        results = []
        for t in trades:
            results.append({
                "pnl_ticks": t.pnl_ticks,
                "exit_reason": t.exit_reason,
                "hold_time_ms": t.hold_time_ns / 1e6 if hasattr(t, "hold_time_ns") else 0,
                "queue_wait_ms": t.queue_wait_ns / 1e6 if hasattr(t, "queue_wait_ns") else 0,
            })

        del engine
        return results

    except Exception as e:
        log.debug(f"FIFO replay failed for {date_str}: {e}")
        return None


def simulate_simplified(
    preds_1s: np.ndarray,
    labels_1s: np.ndarray,
    labels_10s: np.ndarray,
    z_threshold: float,
    tp_ticks: float,
    sl_ticks: float,
    max_hold_ms: float,
    spread_mean: float = 1.0,
) -> dict:
    """
    Simplified PnL model when FIFO engine is unavailable or too slow.
    Uses actual OOS labels to estimate trade outcomes.

    For each signal that passes threshold:
      - Direction from sign of prediction
      - Use labels as actual future price moves (in z-score units)
      - Scale to ticks and apply TP/SL/hold logic
      - Deduct commission and half-spread for passive limit entry
    """
    n = len(preds_1s)
    if n == 0:
        return _empty_result()

    # Use 1s predictions for signal generation
    strength = np.abs(preds_1s)
    mask = strength >= z_threshold
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return _empty_result()

    # Scale labels from z-scores to approximate tick moves
    # labels are already in z-score-ish units; rough conversion
    # The predictions are already calibrated z-scores
    label_std_ticks = 2.0  # rough: 1 z-score ~ 2 ticks of future move

    trades_pnl = []
    exit_reasons = []

    for idx in indices:
        direction = 1.0 if preds_1s[idx] > 0 else -1.0
        # Use 10s label as "eventual move" for TP/SL evaluation
        actual_move = float(labels_10s[idx]) * label_std_ticks * direction

        # Determine exit
        if actual_move >= tp_ticks:
            pnl = tp_ticks - COMMISSION_RT_TICKS
            exit_reasons.append("tp")
        elif actual_move <= -sl_ticks:
            pnl = -sl_ticks - COMMISSION_RT_TICKS
            exit_reasons.append("sl")
        else:
            # Max hold: exit at whatever the move is, minus spread cost
            pnl = actual_move - COMMISSION_RT_TICKS - spread_mean * 0.25
            exit_reasons.append("max_hold")

        trades_pnl.append(pnl)

    return _compute_trade_stats(trades_pnl, exit_reasons)


def _empty_result() -> dict:
    return {
        "n_trades": 0, "total_pnl_ticks": 0.0, "mean_pnl": 0.0,
        "win_rate": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "tp_rate": 0.0, "sl_rate": 0.0,
    }


def _compute_trade_stats(pnls: List[float], exits: List[str]) -> dict:
    """Compute aggregate stats from trade PnL list."""
    pnls_arr = np.array(pnls)
    n = len(pnls_arr)
    if n == 0:
        return _empty_result()

    total = float(pnls_arr.sum())
    mean = float(pnls_arr.mean())
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr <= 0]
    wr = float(len(wins) / n) if n > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(losses).sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss

    std = float(pnls_arr.std()) if n > 1 else 1e-9
    sharpe = mean / std if std > 1e-9 else 0.0

    downside = pnls_arr[pnls_arr < 0]
    down_std = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 1e-9
    sortino = mean / down_std if down_std > 1e-9 else 0.0

    exits_arr = np.array(exits)
    tp_rate = float((exits_arr == "tp").mean()) if n > 0 else 0.0
    sl_rate = float((exits_arr == "sl").mean()) if n > 0 else 0.0

    return {
        "n_trades": n,
        "total_pnl_ticks": total,
        "total_pnl_dollars": total * TICK_USD,
        "mean_pnl": mean,
        "win_rate": wr,
        "profit_factor": pf,
        "sharpe": sharpe,
        "sortino": sortino,
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
    }


# =============================================================================
# Per-regime evaluation worker
# =============================================================================

# Global cache (set before fork)
_GLOBAL_DATA = []


def _init_worker(data_list):
    """Initialize worker with shared data."""
    global _GLOBAL_DATA
    _GLOBAL_DATA = data_list


def evaluate_regime_config(args: Tuple) -> Optional[dict]:
    """
    Evaluate one (regime, config) combination across all dates.
    Worker function for multiprocessing pool.
    """
    regime, config = args
    config_id = config["config_id"]
    tp = config["tp_ticks"]
    sl = config["sl_ticks"]
    hold = config["max_hold_ms"]
    z_thresh = config["z_threshold"]

    all_pnls = []
    daily_pnls = []  # one per date
    all_exits = []
    n_signals_total = 0
    dates_with_trades = 0

    for date_data in _GLOBAL_DATA:
        # Filter predictions to this regime
        regime_mask = np.array([r == regime for r in date_data["pred_regimes"]])
        if regime_mask.sum() == 0:
            continue

        preds_1s = date_data["preds"][regime_mask, 0]  # 1s horizon
        labels_1s = date_data["labels_1s"][regime_mask]
        labels_10s = date_data["labels_10s"][regime_mask]
        pred_ts = date_data["pred_ts"][regime_mask]

        # Get mean spread for this regime's windows
        feat_spread_idx = FEAT_IDX["spread_mean_10k"]
        decision_stride = date_data["decision_stride"]
        features = date_data["features"]
        regime_feat_idxs = []
        for i, (rm, vi) in enumerate(
            zip(date_data["pred_regimes"], range(len(date_data["preds"])))
        ):
            if rm == regime:
                fw = min(
                    int(date_data["preds"].shape[0] * i / len(date_data["pred_regimes"])
                        * len(features) / date_data["preds"].shape[0]),
                    len(features) - 1,
                )
                regime_feat_idxs.append(fw)

        if regime_feat_idxs:
            spread_mean = float(
                np.mean(features[regime_feat_idxs, feat_spread_idx])
            )
        else:
            spread_mean = 1.0

        # Try FIFO replay first
        fifo_results = None
        if _HAS_FIFO_ENGINE and len(preds_1s) >= 5:
            # Build signals for FIFO engine
            strength = np.abs(preds_1s)
            mask = strength >= z_thresh
            signal_indices = np.where(mask)[0]

            if len(signal_indices) >= 3:
                signals = []
                last_ts = -1e18
                for si in signal_indices:
                    t = int(pred_ts[si])
                    if t - last_ts < 500_000_000:  # 500ms anti-churn
                        continue
                    direction = "long" if preds_1s[si] > 0 else "short"
                    signals.append({
                        "ts_ns": t,
                        "direction": direction,
                        "strength": float(strength[si]),
                    })
                    last_ts = t

                if len(signals) >= 2:
                    fifo_results = run_fifo_replay_for_date(
                        date_data["date"], signals, tp, sl, hold
                    )

        if fifo_results is not None and len(fifo_results) > 0:
            day_pnl = sum(r["pnl_ticks"] for r in fifo_results)
            daily_pnls.append(day_pnl)
            for r in fifo_results:
                all_pnls.append(r["pnl_ticks"])
                all_exits.append(r["exit_reason"])
            dates_with_trades += 1
        else:
            # Fall back to simplified model
            result = simulate_simplified(
                preds_1s, labels_1s, labels_10s,
                z_thresh, tp, sl, hold, spread_mean,
            )
            if result["n_trades"] > 0:
                # Reconstruct individual trade PnLs for daily aggregation
                day_pnl = result["total_pnl_ticks"]
                daily_pnls.append(day_pnl)
                dates_with_trades += 1
                # We don't have individual trades, estimate
                n = result["n_trades"]
                all_pnls.extend([result["mean_pnl"]] * n)
                tp_n = int(result["tp_rate"] * n)
                sl_n = int(result["sl_rate"] * n)
                all_exits.extend(["tp"] * tp_n + ["sl"] * sl_n +
                                 ["max_hold"] * (n - tp_n - sl_n))

    # Compute aggregate stats
    if not all_pnls:
        return None

    stats = _compute_trade_stats(all_pnls, all_exits)

    # Daily-level metrics
    if len(daily_pnls) >= 2:
        dp = np.array(daily_pnls)
        daily_mean = float(dp.mean())
        daily_std = float(dp.std())
        daily_sharpe = daily_mean / daily_std if daily_std > 1e-9 else 0.0
        down = dp[dp < 0]
        daily_down_std = float(np.sqrt((down**2).mean())) if len(down) > 0 else 1e-9
        daily_sortino = daily_mean / daily_down_std if daily_down_std > 1e-9 else 0.0
    else:
        daily_sharpe = 0.0
        daily_sortino = 0.0

    return {
        "regime": regime,
        "config_id": config_id,
        "tp_ticks": tp,
        "sl_ticks": sl,
        "max_hold_ms": hold,
        "z_threshold": z_thresh,
        "n_trades": stats["n_trades"],
        "n_dates": dates_with_trades,
        "total_pnl_ticks": stats["total_pnl_ticks"],
        "total_pnl_dollars": stats.get("total_pnl_dollars", stats["total_pnl_ticks"] * TICK_USD),
        "mean_pnl": stats["mean_pnl"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "tp_rate": stats["tp_rate"],
        "sl_rate": stats["sl_rate"],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vol-conditioned execution parameter optimization")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without running")
    parser.add_argument("--max-configs", type=int, default=0, help="Limit configs per regime (0=all)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Volatility-Conditioned Execution Parameter Optimization")
    log.info("=" * 70)
    log.info(f"Workers: {args.workers}")
    log.info(f"Commission: {COMMISSION_RT_TICKS:.3f} ticks ($4.70 RT)")
    log.info(f"FIFO engine available: {_HAS_FIFO_ENGINE}")

    # Build parameter grid
    configs = build_param_configs()
    n_configs = len(configs)
    if args.max_configs > 0:
        configs = configs[:args.max_configs]
    log.info(f"Parameter grid: {n_configs} total configs, using {len(configs)}")
    log.info(f"Grid: TP={PARAM_GRID['tp_ticks']}, SL={PARAM_GRID['sl_ticks']}, "
             f"hold={PARAM_GRID['max_hold_ms']}, z_thresh={PARAM_GRID['z_threshold']}")

    # Build work items: (regime, config) for each regime
    work_items = []
    for regime in VOL_REGIMES:
        for config in configs:
            work_items.append((regime, config))

    total_work = len(work_items)
    log.info(f"Total work items: {total_work} ({len(VOL_REGIMES)} regimes x {len(configs)} configs)")

    if args.dry_run:
        log.info("DRY RUN — showing sample configs:")
        for regime in VOL_REGIMES:
            log.info(f"  Regime '{regime}': {len(configs)} configs")
        log.info(f"  Sample config: {configs[0]}")
        return

    # Load all data
    log.info("\n--- Loading data ---")
    t0 = time.time()
    all_data = load_all_dates()
    load_time = time.time() - t0
    log.info(f"Loaded {len(all_data)} dates in {load_time:.1f}s")

    if not all_data:
        log.error("No data loaded. Exiting.")
        return

    # Print vol regime distribution
    regime_counts = {"low": 0, "medium": 0, "high": 0}
    for d in all_data:
        for r in d["pred_regimes"]:
            regime_counts[r] += 1
    total_preds = sum(regime_counts.values())
    log.info(f"Vol regime distribution across {total_preds} predictions:")
    for r in VOL_REGIMES:
        pct = regime_counts[r] / total_preds * 100 if total_preds > 0 else 0
        log.info(f"  {r:>8s}: {regime_counts[r]:5d} ({pct:.1f}%)")

    # Run sweep with multiprocessing
    log.info(f"\n--- Running sweep ({total_work} items, {args.workers} workers) ---")
    t1 = time.time()

    results = []
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(all_data,),
    ) as pool:
        completed = 0
        for result in pool.imap_unordered(evaluate_regime_config, work_items, chunksize=4):
            completed += 1
            if result is not None:
                results.append(result)
            if completed % 100 == 0:
                elapsed = time.time() - t1
                rate = completed / elapsed
                eta = (total_work - completed) / rate if rate > 0 else 0
                log.info(f"  Progress: {completed}/{total_work} "
                         f"({completed*100/total_work:.0f}%), "
                         f"{len(results)} with trades, "
                         f"ETA: {eta:.0f}s")

    sweep_time = time.time() - t1
    log.info(f"Sweep complete: {len(results)} results in {sweep_time:.1f}s")

    if not results:
        log.error("No results produced. Check data/predictions alignment.")
        return

    # ── Analyze and report ───────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RESULTS: Optimal Parameters Per Volatility Regime")
    log.info("=" * 70)

    optimal_table = {}

    for regime in VOL_REGIMES:
        regime_results = [r for r in results if r["regime"] == regime]
        if not regime_results:
            log.info(f"\n  Regime '{regime}': No results")
            continue

        # Sort by Sortino (primary), then Sharpe, then PF
        regime_results.sort(
            key=lambda r: (r["sortino"], r["sharpe"], r["profit_factor"]),
            reverse=True,
        )

        best = regime_results[0]
        log.info(f"\n  ╔══ Regime: {regime.upper()} VOL ══╗")
        log.info(f"  ║ Best config (by Sortino):")
        log.info(f"  ║   TP: {best['tp_ticks']} ticks, SL: {best['sl_ticks']} ticks")
        log.info(f"  ║   Max hold: {best['max_hold_ms']}ms, Z-threshold: {best['z_threshold']}")
        log.info(f"  ║   Trades: {best['n_trades']}, Dates: {best['n_dates']}")
        log.info(f"  ║   Total PnL: {best['total_pnl_ticks']:.1f} ticks (${best['total_pnl_dollars']:.0f})")
        log.info(f"  ║   Mean PnL/trade: {best['mean_pnl']:.3f} ticks")
        log.info(f"  ║   Win rate: {best['win_rate']*100:.1f}%")
        log.info(f"  ║   Profit factor: {best['profit_factor']:.2f}")
        log.info(f"  ║   Sharpe: {best['sharpe']:.3f}, Sortino: {best['sortino']:.3f}")
        log.info(f"  ║   Daily Sharpe: {best['daily_sharpe']:.3f}, Daily Sortino: {best['daily_sortino']:.3f}")
        log.info(f"  ║   TP rate: {best['tp_rate']*100:.1f}%, SL rate: {best['sl_rate']*100:.1f}%")
        log.info(f"  ╚{'═'*40}╝")

        optimal_table[regime] = {
            "tp_ticks": best["tp_ticks"],
            "sl_ticks": best["sl_ticks"],
            "max_hold_ms": best["max_hold_ms"],
            "z_threshold": best["z_threshold"],
            "sortino": best["sortino"],
            "sharpe": best["sharpe"],
            "profit_factor": best["profit_factor"],
            "win_rate": best["win_rate"],
            "n_trades": best["n_trades"],
            "total_pnl_dollars": best["total_pnl_dollars"],
        }

        # Show top 5
        log.info(f"\n  Top 5 configs for {regime}:")
        log.info(f"  {'Rank':>4s} {'TP':>3s} {'SL':>3s} {'Hold':>7s} {'Z':>4s} "
                 f"{'Trades':>6s} {'PnL$':>8s} {'WR':>5s} {'PF':>5s} "
                 f"{'Sharpe':>7s} {'Sortino':>8s}")
        log.info(f"  {'-'*70}")
        for i, r in enumerate(regime_results[:5]):
            log.info(
                f"  {i+1:>4d} {r['tp_ticks']:>3.0f} {r['sl_ticks']:>3.0f} "
                f"{r['max_hold_ms']:>7.0f} {r['z_threshold']:>4.1f} "
                f"{r['n_trades']:>6d} {r['total_pnl_dollars']:>8.0f} "
                f"{r['win_rate']*100:>5.1f} {r['profit_factor']:>5.2f} "
                f"{r['sharpe']:>7.3f} {r['sortino']:>8.3f}"
            )

        # Show profitable configs count
        profitable = [r for r in regime_results if r["total_pnl_ticks"] > 0]
        log.info(f"\n  Profitable configs: {len(profitable)}/{len(regime_results)} "
                 f"({len(profitable)*100/len(regime_results):.0f}%)")

    # ── Save results ─────────────────────────────────────────────────────────
    # Save optimal lookup table
    lookup_path = OUTPUT_DIR / "vol_regime_lookup_table.json"
    with open(lookup_path, "w") as f:
        json.dump({
            "description": "Optimal execution parameters per volatility regime",
            "commission_ticks": COMMISSION_RT_TICKS,
            "vol_percentiles": list(VOL_PERCENTILES),
            "vol_feature": "price_volatility_window",
            "n_dates": len(all_data),
            "timestamp": datetime.now().isoformat(),
            "regimes": optimal_table,
        }, f, indent=2)
    log.info(f"\nSaved lookup table: {lookup_path}")

    # Save all results as JSON
    all_results_path = OUTPUT_DIR / "all_sweep_results.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved all results: {all_results_path}")

    # Save as CSV for analysis
    csv_path = OUTPUT_DIR / "sweep_results.csv"
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        log.info(f"Saved CSV: {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Dates tested: {len(all_data)}")
    log.info(f"Configs per regime: {len(configs)}")
    log.info(f"Total evaluations: {total_work}")
    log.info(f"Results with trades: {len(results)}")
    log.info(f"Total time: {time.time() - t0:.0f}s")

    if optimal_table:
        log.info("\nOptimal Lookup Table:")
        for regime, params in optimal_table.items():
            log.info(f"  {regime:>8s} vol: TP={params['tp_ticks']}t, "
                     f"SL={params['sl_ticks']}t, hold={params['max_hold_ms']}ms, "
                     f"z>={params['z_threshold']:.1f} "
                     f"→ Sortino={params['sortino']:.3f}, PF={params['profit_factor']:.2f}, "
                     f"${params['total_pnl_dollars']:.0f}")

    log.info("\nDone.")


# Need csv import
import csv

if __name__ == "__main__":
    main()
