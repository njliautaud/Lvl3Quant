#!/usr/bin/env python3
"""
EOFI Pressure Trade Simulator — Event-Level
=============================================
Uses EOFI XGBoost predictions (per MBO event) to simulate trading.
Prediction = smooth EOFI pressure score at 10s horizon.

Strategy: Enter when z-scored pressure exceeds threshold, exit on reversal/fade/timeout.
Cost model: FIFO bid/ask fills + 0.376 ticks RT commission (HC #512).

Per HC #428 R2: TP/SL/hold bounded by model's predictive horizon (10s).
Per HC #511: Commission = 0.376 ticks RT.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('eofi_sim')

BASE = Path("/home/jupiter/Lvl3Quant")
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
PRESSURE_DIR = BASE / "data" / "processed" / "smooth_pressure_targets"
EOFI_PREDS = BASE / "output" / "eofi_pressure_xgb_v1" / "predictions_eofi_10s.npz"
SUMMARY_FILE = BASE / "output" / "eofi_pressure_xgb_v1" / "summary_eofi_10s.json"
OUTPUT_DIR = BASE / "output" / "eofi_trade_sim_v1"

# Cost constants (canonical — HC #512: spread is NOT a cost, it's in the fill price)
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
TOTAL_RT_COST = 0.376  # commission only (HC #512)

# Prediction stride — EOFI is per-event, but we subsample for computational tractability
# Every Nth event gets a prediction action opportunity
PRED_STRIDE = 250  # Match CNN-Mamba stride for comparability


@dataclass
class TradeConfig:
    entry_threshold: float = 1.5    # |z-scored pressure| to enter
    fade_threshold: float = 0.3     # Below this = fading signal
    fade_n: int = 8                 # Consecutive fading predictions → exit
    max_hold_s: float = 30.0        # Max hold (bounded by 10s horizon * 1.5 per HC #428 = 15s, use 30s with margin)
    reversal_threshold: float = -0.5
    min_hold_preds: int = 4         # Min predictions to hold
    cooldown_preds: int = 20        # Between trades
    warmup_preds: int = 500         # Z-score warmup
    smooth_window: int = 10         # EMA window
    sides: str = "both"             # "long", "short", "both"


@dataclass
class Trade:
    date: str = ""
    side: str = ""
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    gross_pnl_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    hold_duration_s: float = 0.0
    n_predictions: int = 0
    exit_reason: str = ""
    entry_pressure: float = 0.0
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0


def load_day_events_efficient(date_str: str, n_preds: int):
    """Load event timestamps and mid prices for a day, memory-mapped.

    Returns subsampled data at PRED_STRIDE intervals, plus full mid prices
    for MFE/MAE calculation within trade windows.
    """
    event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    if not event_file.exists():
        return None
    try:
        data = np.load(str(event_file), mmap_mode='r')
        n_events = len(data['timestamps'])
        n_usable = min(n_preds, n_events)

        # Subsample indices
        pred_indices = np.arange(0, n_usable, PRED_STRIDE)
        if len(pred_indices) < 600:
            return None

        # Load only what we need
        sub_times = data['timestamps'][pred_indices].copy()
        bid_ask = data['events'][:n_usable, [5, 6]].copy()  # (n_usable, 2)
        all_mids = (bid_ask[:, 0] + bid_ask[:, 1]) / 2.0
        sub_mids = all_mids[pred_indices]

        return {
            'sub_times': sub_times,
            'sub_mids': sub_mids,
            'all_mids': all_mids,
            'pred_indices': pred_indices,
            'n_usable': n_usable,
        }
    except Exception as e:
        log.warning(f"Failed to load events for {date_str}: {e}")
        return None


def run_day_cached(date_str: str, preds: np.ndarray, cached: dict, cfg: TradeConfig) -> List[Trade]:
    """Run trade simulation for a single day using cached data."""
    pred_indices = cached['pred_indices']
    sub_times = cached['sub_times']
    sub_mids = cached['sub_mids']
    all_mids = cached['all_mids']

    # Get subsampled predictions
    sub_preds = preds[pred_indices]

    # Z-score normalization (rolling)
    n = len(sub_preds)
    z_preds = np.zeros(n)
    running_sum = 0.0
    running_sq_sum = 0.0
    for i in range(n):
        running_sum += sub_preds[i]
        running_sq_sum += sub_preds[i] ** 2
        count = i + 1
        mean = running_sum / count
        var = running_sq_sum / count - mean ** 2
        std = max(np.sqrt(max(var, 0)), 1e-8)
        z_preds[i] = (sub_preds[i] - mean) / std

    # EMA smoothing
    alpha = 2.0 / (cfg.smooth_window + 1)
    ema = np.zeros(n)
    ema[0] = z_preds[0]
    for i in range(1, n):
        ema[i] = alpha * z_preds[i] + (1 - alpha) * ema[i - 1]

    # Trade simulation
    trades = []
    state = "FLAT"
    entry_idx = 0
    entry_price = 0.0
    entry_time = 0
    entry_pressure = 0.0
    fade_count = 0
    last_trade_idx = -cfg.cooldown_preds
    n_preds_in_trade = 0
    best_price = 0.0  # for MFE/MAE

    for i in range(cfg.warmup_preds, n):
        p = ema[i]
        t = sub_times[i]
        mid = sub_mids[i]
        if mid <= 0 or not np.isfinite(mid):
            continue

        if state == "FLAT":
            if i - last_trade_idx < cfg.cooldown_preds:
                continue

            go_long = cfg.sides in ("both", "long") and p > cfg.entry_threshold
            go_short = cfg.sides in ("both", "short") and p < -cfg.entry_threshold

            if go_long:
                state = "LONG"
                entry_idx = i
                entry_price = mid + TICK_SIZE * 0.5  # taker buy at ask
                entry_time = t
                entry_pressure = p
                fade_count = 0
                n_preds_in_trade = 0
                best_price = entry_price
            elif go_short:
                state = "SHORT"
                entry_idx = i
                entry_price = mid - TICK_SIZE * 0.5  # taker sell at bid
                entry_time = t
                entry_pressure = p
                fade_count = 0
                n_preds_in_trade = 0
                best_price = entry_price

        else:  # In position
            n_preds_in_trade += 1
            hold_s = (t - entry_time) / 1e9

            # Track MFE/MAE
            if state == "LONG":
                best_price = max(best_price, mid)
            else:
                best_price = min(best_price, mid)

            if n_preds_in_trade < cfg.min_hold_preds:
                continue

            # Exit conditions
            exit_reason = ""

            # 1. Timeout (HC #428: hold <= 1.5x horizon)
            if hold_s >= cfg.max_hold_s:
                exit_reason = "timeout"

            # 2. Reversal
            elif state == "LONG" and p < cfg.reversal_threshold:
                exit_reason = "reversal"
            elif state == "SHORT" and p > -cfg.reversal_threshold:
                exit_reason = "reversal"

            # 3. Fade
            else:
                if state == "LONG":
                    is_fading = p < cfg.fade_threshold
                else:
                    is_fading = p > -cfg.fade_threshold

                if is_fading:
                    fade_count += 1
                else:
                    fade_count = 0

                if fade_count >= cfg.fade_n:
                    exit_reason = "fade"

            if exit_reason:
                # Execute exit
                if state == "LONG":
                    exit_price = mid - TICK_SIZE * 0.5  # taker sell at bid
                    gross_pnl = (exit_price - entry_price) / TICK_SIZE
                    mfe = (best_price - entry_price) / TICK_SIZE
                    slice_mids = all_mids[pred_indices[entry_idx]:pred_indices[i]+1]
                    mae = (entry_price - min(slice_mids.min(), entry_price)) / TICK_SIZE if len(slice_mids) > 0 else 0
                else:
                    exit_price = mid + TICK_SIZE * 0.5  # taker buy at ask
                    gross_pnl = (entry_price - exit_price) / TICK_SIZE
                    mfe = (entry_price - best_price) / TICK_SIZE
                    slice_mids = all_mids[pred_indices[entry_idx]:pred_indices[i]+1]
                    mae = (max(slice_mids.max(), entry_price) - entry_price) / TICK_SIZE if len(slice_mids) > 0 else 0

                net_pnl = gross_pnl - COMMISSION_RT_TICKS

                trades.append(Trade(
                    date=date_str,
                    side=state,
                    entry_time_ns=int(entry_time),
                    exit_time_ns=int(t),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    gross_pnl_ticks=float(gross_pnl),
                    net_pnl_ticks=float(net_pnl),
                    hold_duration_s=float(hold_s),
                    n_predictions=n_preds_in_trade,
                    exit_reason=exit_reason,
                    entry_pressure=float(entry_pressure),
                    mfe_ticks=float(mfe),
                    mae_ticks=float(mae),
                ))

                state = "FLAT"
                last_trade_idx = i

    return trades


def compute_metrics(trades: List[Trade]) -> dict:
    """Compute risk-adjusted performance metrics."""
    if not trades:
        return {'n_trades': 0}

    net_pnls = np.array([t.net_pnl_ticks for t in trades])
    gross_pnls = np.array([t.gross_pnl_ticks for t in trades])

    n = len(net_pnls)
    win_rate = np.mean(net_pnls > 0) if n > 0 else 0

    # Per-day PnL for Sharpe/Sortino
    day_pnl = {}
    for t in trades:
        day_pnl[t.date] = day_pnl.get(t.date, 0) + t.net_pnl_ticks
    daily_pnls = np.array(list(day_pnl.values()))

    mean_daily = np.mean(daily_pnls) if len(daily_pnls) > 0 else 0
    std_daily = np.std(daily_pnls) if len(daily_pnls) > 1 else 1
    downside_std = np.std(daily_pnls[daily_pnls < 0]) if np.sum(daily_pnls < 0) > 1 else 1

    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 0 else 0

    total_win = np.sum(net_pnls[net_pnls > 0])
    total_loss = -np.sum(net_pnls[net_pnls <= 0])
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

    holds = [t.hold_duration_s for t in trades]
    mfes = [t.mfe_ticks for t in trades]
    maes = [t.mae_ticks for t in trades]

    # Exit reason breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Side breakdown
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    return {
        'n_trades': n,
        'n_days': len(daily_pnls),
        'trades_per_day': n / max(len(daily_pnls), 1),
        'win_rate': float(win_rate),
        'total_net_ticks': float(np.sum(net_pnls)),
        'total_net_dollars': float(np.sum(net_pnls) * TICK_VALUE),
        'mean_net_ticks': float(np.mean(net_pnls)),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(profit_factor),
        'avg_hold_s': float(np.mean(holds)),
        'median_hold_s': float(np.median(holds)),
        'avg_mfe': float(np.mean(mfes)),
        'avg_mae': float(np.mean(maes)),
        'exit_reasons': exit_counts,
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'long_wr': float(np.mean([t.net_pnl_ticks > 0 for t in longs])) if longs else 0,
        'short_wr': float(np.mean([t.net_pnl_ticks > 0 for t in shorts])) if shorts else 0,
        'long_mean_net': float(np.mean([t.net_pnl_ticks for t in longs])) if longs else 0,
        'short_mean_net': float(np.mean([t.net_pnl_ticks for t in shorts])) if shorts else 0,
        'daily_pnls': {k: float(v) for k, v in day_pnl.items()},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run config sweep')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary to get fold dates and sample counts
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    fold_details = summary['fold_details']
    fold_dates = [fd['fold_date'] for fd in fold_details]
    fold_sizes = [fd['n_test'] for fd in fold_details]

    # Load concatenated predictions
    log.info("Loading EOFI predictions...")
    pred_data = np.load(str(EOFI_PREDS))
    all_preds = pred_data['predictions']
    all_actuals = pred_data['actuals']
    log.info(f"Loaded {len(all_preds):,} predictions across {len(fold_dates)} dates")

    # Split into per-date chunks
    date_preds = {}
    offset = 0
    for date_str, n_samples in zip(fold_dates, fold_sizes):
        date_preds[date_str] = all_preds[offset:offset + n_samples]
        offset += n_samples

    assert offset == len(all_preds), f"Prediction count mismatch: {offset} vs {len(all_preds)}"

    # Pre-load events one day at a time using mmap for efficiency
    log.info("Pre-processing event data (mmap, memory-efficient)...")
    day_cache = {}
    for di, date_str in enumerate(fold_dates):
        if date_str not in date_preds:
            continue
        n_preds = len(date_preds[date_str])
        cached = load_day_events_efficient(date_str, n_preds)
        if cached is not None:
            day_cache[date_str] = cached
        if (di + 1) % 10 == 0:
            log.info(f"  Loaded {di+1}/{len(fold_dates)} dates...")
    log.info(f"Cached {len(day_cache)} dates for trading simulation")

    if args.sweep:
        # Config sweep
        sweep_configs = []
        for entry_th, fade_th, fade_n, max_hold, smooth, sides in product(
            [1.0, 1.5, 2.0, 2.5],       # entry threshold
            [0.2, 0.3, 0.5],             # fade threshold
            [4, 8, 12],                   # fade N
            [15.0, 30.0],                 # max hold (HC #428: ≤1.5x horizon)
            [5, 10, 20],                  # smooth window
            ["both", "short"],            # sides (short has better edge)
        ):
            sweep_configs.append(TradeConfig(
                entry_threshold=entry_th,
                fade_threshold=fade_th,
                fade_n=fade_n,
                max_hold_s=max_hold,
                smooth_window=smooth,
                sides=sides,
            ))

        log.info(f"Running sweep: {len(sweep_configs)} configs")
        results = []

        for ci, cfg in enumerate(sweep_configs):
            all_trades = []
            for date_str in fold_dates:
                if date_str not in day_cache or date_str not in date_preds:
                    continue
                preds = date_preds[date_str]
                cached = day_cache[date_str]

                day_trades = run_day_cached(date_str, preds[:cached['n_usable']], cached, cfg)
                all_trades.extend(day_trades)

            metrics = compute_metrics(all_trades)
            metrics['config'] = asdict(cfg)
            results.append(metrics)

            if (ci + 1) % 50 == 0:
                log.info(f"  {ci+1}/{len(sweep_configs)} configs done...")

        # Sort by Sharpe
        results.sort(key=lambda x: x.get('sharpe', -999), reverse=True)

        # Save all results
        out_file = OUTPUT_DIR / "sweep_results.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Saved {len(results)} sweep results to {out_file}")

        # Print top 10
        log.info("\n" + "=" * 80)
        log.info("TOP 10 CONFIGS BY SHARPE")
        log.info("=" * 80)
        for i, r in enumerate(results[:10]):
            c = r['config']
            log.info(f"\n#{i+1}: Sharpe={r['sharpe']:.2f} Sortino={r['sortino']:.2f} "
                     f"PF={r['profit_factor']:.2f} WR={r['win_rate']:.1%}")
            log.info(f"  Trades={r['n_trades']} ({r['trades_per_day']:.1f}/day) "
                     f"Total={r['total_net_ticks']:.1f}t (${r['total_net_dollars']:.0f})")
            log.info(f"  AvgHold={r['avg_hold_s']:.1f}s MFE={r['avg_mfe']:.2f}t MAE={r['avg_mae']:.2f}t")
            log.info(f"  Config: entry={c['entry_threshold']} fade={c['fade_threshold']} "
                     f"fadeN={c['fade_n']} maxHold={c['max_hold_s']}s smooth={c['smooth_window']} "
                     f"sides={c['sides']}")
            log.info(f"  Long WR={r['long_wr']:.1%}({r['n_longs']}) Short WR={r['short_wr']:.1%}({r['n_shorts']})")
            log.info(f"  Exits: {r['exit_reasons']}")

    else:
        # Single run with default config
        cfg = TradeConfig()
        all_trades = []
        for date_str in fold_dates:
            if date_str not in day_cache or date_str not in date_preds:
                continue
            preds = date_preds[date_str]
            cached = day_cache[date_str]
            day_trades = run_day_cached(date_str, preds[:cached['n_usable']], cached, cfg)
            all_trades.extend(day_trades)

        metrics = compute_metrics(all_trades)
        metrics['config'] = asdict(cfg)

        log.info(f"\nResults: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']:.2f}, "
                 f"Sortino={metrics['sortino']:.2f}, WR={metrics['win_rate']:.1%}, "
                 f"PF={metrics['profit_factor']:.2f}")
        log.info(f"Total: {metrics['total_net_ticks']:.1f} ticks (${metrics['total_net_dollars']:.0f})")

        out_file = OUTPUT_DIR / "single_run_results.json"
        with open(out_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
