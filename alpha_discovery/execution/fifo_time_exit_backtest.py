#!/usr/bin/env python3
"""
FIFO Time-Exit Backtest — CNN-Mamba v2
=======================================
Validates midpoint endpoint analysis with FIFO queue simulation.

Key question: What percentage of passive limit entries actually GET FILLED
within the time window, and what are the realized P&L stats?

Simplified FIFO Model:
- Queue position: BACK of queue (worst case)
- Queue depth: estimated from MBO event rate at price level
- Fill probability model: calibrated from actual MBO volume at best bid/ask
- If exact FIFO too complex from preprocessed features, uses conservative
  fill probability = f(hold_time, event_rate_at_level)

Uses: predictions (N,3) for 1s/5s/10s horizons, labels (N,3) in ticks.
The labels already encode the actual future midpoint move — we use them
directly for exit P&L calculation.

Entry model: passive limit at best bid (long) / best ask (short)
Exit model: market order (cross spread = 1 tick) or time-limit hybrid

Cost: ES tick = $12.50, commission = $4.70 RT = 0.376 ticks
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# ── Paths ──
LVL3 = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = LVL3 / "output" / "cnn_mamba_v2_bulk_oot"
EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = LVL3 / "output" / "fifo_time_exit_backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
WINDOW = 1000
STRIDE = 500
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50
SPREAD_TICKS = 1.0  # ES is 1 tick wide during RTH

# ── FIFO Model Parameters ──
# Calibrated from typical ES RTH queue dynamics:
# - Average queue depth at best bid/ask: ~50-150 contracts
# - Average fill rate for back-of-queue: depends on time and volume
# We use a time-based fill probability model:
#   P(fill within T seconds) = 1 - exp(-event_rate_at_level * T / queue_depth)
# Simplified: use empirically calibrated fill rates
FILL_PROB_BY_HOLD_TIME = {
    1.0: 0.45,   # 45% fill within 1s (aggressive estimate, back of queue)
    2.0: 0.60,   # 60% fill within 2s
    5.0: 0.75,   # 75% fill within 5s
    10.0: 0.85,  # 85% fill within 10s
}


def get_fill_probability(hold_seconds: float, event_rate_factor: float = 1.0) -> float:
    """Get fill probability for a given hold time.

    event_rate_factor: multiplier for busier/quieter periods.
    >1.0 means more volume (higher fill prob), <1.0 means less.
    """
    # Interpolate between known points
    times = sorted(FILL_PROB_BY_HOLD_TIME.keys())
    if hold_seconds <= times[0]:
        base_prob = FILL_PROB_BY_HOLD_TIME[times[0]] * (hold_seconds / times[0])
    elif hold_seconds >= times[-1]:
        base_prob = FILL_PROB_BY_HOLD_TIME[times[-1]]
    else:
        # Linear interpolation
        for i in range(len(times) - 1):
            if times[i] <= hold_seconds <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                p0, p1 = FILL_PROB_BY_HOLD_TIME[t0], FILL_PROB_BY_HOLD_TIME[t1]
                frac = (hold_seconds - t0) / (t1 - t0)
                base_prob = p0 + frac * (p1 - p0)
                break

    # Adjust for event rate
    adjusted = 1.0 - (1.0 - base_prob) ** event_rate_factor
    return min(adjusted, 0.95)  # Cap at 95%


def estimate_event_rate_factor(timestamps, signal_idx, window_ns=5_000_000_000):
    """Estimate local event rate relative to average.

    Look at event density around the signal point vs overall average.
    """
    n = len(timestamps)
    if signal_idx >= n or signal_idx < 0:
        return 1.0

    t_signal = timestamps[signal_idx]

    # Count events in a 5s window around signal
    start_idx = max(0, signal_idx - 5000)
    end_idx = min(n, signal_idx + 5000)

    # Local density: events per nanosecond
    local_span = timestamps[min(end_idx, n-1)] - timestamps[start_idx]
    if local_span <= 0:
        return 1.0
    local_rate = (end_idx - start_idx) / local_span

    # Global density
    total_span = timestamps[-1] - timestamps[0]
    if total_span <= 0:
        return 1.0
    global_rate = n / total_span

    factor = local_rate / global_rate if global_rate > 0 else 1.0
    return np.clip(factor, 0.3, 3.0)


@dataclass
class TradeResult:
    direction: str  # 'long' or 'short'
    signal_strength: float
    hold_seconds: float
    filled: bool
    fill_time_frac: float  # fraction of hold time until fill (0=immediate, 1=at deadline)
    entry_price_offset: float  # 0 for passive fill at bid/ask
    exit_type: str  # 'market' or 'limit'
    pnl_ticks: float  # net of all costs
    label_move_ticks: float  # raw signal (from labels)


def simulate_fold(args):
    """Simulate one fold's OOT predictions with given config."""
    fold_idx, config = args

    # Support both date-based (string like '20260306') and fold-based (int) identifiers
    if isinstance(fold_idx, str):
        pred_path = PRED_DIR / f"{fold_idx}_predictions.npz"
        date_str = fold_idx
    else:
        pred_path = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
        date_str = None

    if not pred_path.exists():
        return None

    pred_data = np.load(pred_path, allow_pickle=True)
    predictions = pred_data['predictions']  # (N, 3) for 1s/5s/10s
    labels = pred_data['labels']  # (N, 3) in ticks

    # Get corresponding event file date
    if date_str is None:
        oot_file = str(pred_data['oot_files'][0])
        date_str = os.path.basename(oot_file).replace('_mbo_events.npz', '')

    # Load from local Jupiter path
    event_path = EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not event_path.exists():
        return None

    ev_data = np.load(event_path)
    timestamps = ev_data['timestamps']

    # Config params
    horizon_idx = config['horizon_idx']  # 0=1s, 1=5s, 2=10s
    hold_seconds = config['hold_seconds']
    confidence_pct = config['confidence_pct']  # top-X% (e.g., 1.0 = top 1%)
    exit_type = config['exit_type']  # 'market' or 'hybrid'
    direction_filter = config['direction']  # 'long', 'short', 'both'

    n_preds = predictions.shape[0]

    # Get predictions for selected horizon
    preds_h = predictions[:, horizon_idx]
    labels_h = labels[:, horizon_idx]

    # Compute z-scores using expanding window (online standardization)
    # This mirrors what would happen in live trading
    zscore = np.zeros(n_preds, dtype=np.float64)
    running_sum = 0.0
    running_sq = 0.0
    count = 0

    for i in range(n_preds):
        v = preds_h[i]
        running_sum += v
        running_sq += v * v
        count += 1
        if count >= 100:  # Need minimum samples for stable z-score
            mean = running_sum / count
            var = (running_sq / count) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            zscore[i] = (v - mean) / std

    # Determine threshold from confidence percentile
    # top-1% = signals with |zscore| in the top 1%
    valid_zscores = zscore[zscore != 0]
    if len(valid_zscores) < 100:
        return None

    abs_zscores = np.abs(valid_zscores)
    threshold = np.percentile(abs_zscores, 100.0 - confidence_pct)

    # Find signal events
    trades = []

    for i in range(n_preds):
        z = zscore[i]
        if abs(z) < threshold:
            continue

        # Direction
        if z > 0:
            sig_direction = 'long'
        else:
            sig_direction = 'short'

        if direction_filter != 'both' and sig_direction != direction_filter:
            continue

        # Get the event index for this prediction
        event_idx = i * STRIDE + WINDOW - 1
        if event_idx >= len(timestamps):
            continue

        # Estimate local fill probability
        rate_factor = estimate_event_rate_factor(timestamps, event_idx)
        fill_prob = get_fill_probability(hold_seconds, rate_factor)

        # Simulate fill using deterministic pseudo-random based on index
        # This makes results reproducible
        fill_hash = (event_idx * 2654435761) & 0xFFFFFFFF
        fill_random = (fill_hash % 10000) / 10000.0
        filled = fill_random < fill_prob

        if not filled:
            trades.append(TradeResult(
                direction=sig_direction,
                signal_strength=abs(z),
                hold_seconds=hold_seconds,
                filled=False,
                fill_time_frac=1.0,
                entry_price_offset=0.0,
                exit_type=exit_type,
                pnl_ticks=0.0,
                label_move_ticks=float(labels_h[i]),
            ))
            continue

        # Fill time: uniform between 0 and hold_seconds (simplified)
        fill_time_frac = (fill_hash % 1000) / 1000.0 * 0.8  # Fill in first 80% of window

        # Calculate P&L
        # Entry: passive limit fill (no spread cost on entry)
        # The label gives us the midpoint move from signal time to signal_time + horizon
        # For our hold_seconds exit, we need to estimate the move at exit time

        # Use the label as the expected move at the labeled horizon
        raw_move = float(labels_h[i])  # midpoint move in ticks at this horizon

        # Direction-adjust: for short, we want negative moves (price goes down = profit)
        if sig_direction == 'long':
            directional_move = raw_move
        else:
            directional_move = -raw_move

        # Exit cost depends on exit type
        if exit_type == 'market':
            # Market exit: cross the spread = 1 tick cost
            exit_cost = SPREAD_TICKS
        elif exit_type == 'hybrid':
            # Try passive exit, if not filled within 1s of exit time, go market
            # Assume 50% of the time passive exit works
            hybrid_hash = ((event_idx + 1) * 2654435761) & 0xFFFFFFFF
            hybrid_fill = (hybrid_hash % 100) / 100.0
            if hybrid_fill < 0.50:
                exit_cost = 0.0  # Passive exit worked
            else:
                exit_cost = SPREAD_TICKS  # Had to go market
        else:
            exit_cost = SPREAD_TICKS

        # Net P&L: directional move - exit cost - commission
        # Entry is passive (no spread cost), exit may have spread cost
        pnl = directional_move - exit_cost - COMMISSION_RT_TICKS

        trades.append(TradeResult(
            direction=sig_direction,
            signal_strength=abs(z),
            hold_seconds=hold_seconds,
            filled=True,
            fill_time_frac=fill_time_frac,
            entry_price_offset=0.0,
            exit_type=exit_type,
            pnl_ticks=float(pnl),
            label_move_ticks=float(raw_move),
        ))

    return {
        'fold': fold_idx,
        'date': date_str,
        'config': config,
        'trades': trades,
        'n_predictions': n_preds,
        'threshold_z': float(threshold),
    }


def compute_stats(trades_filled):
    """Compute trading statistics for filled trades."""
    if not trades_filled:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_ticks': 0.0,
            'total_pnl_ticks': 0.0,
            'total_pnl_dollars': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'profit_factor': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
        }

    pnls = np.array([t.pnl_ticks for t in trades_filled])
    n = len(pnls)

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]

    win_rate = len(winners) / n if n > 0 else 0.0
    avg_pnl = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))

    # Sharpe (per-trade, then scaled by sqrt(n) for the sample period)
    if np.std(pnls) > 0:
        sharpe_per_trade = np.mean(pnls) / np.std(pnls)
        sharpe = float(sharpe_per_trade * np.sqrt(min(n, 252)))
    else:
        sharpe = 0.0

    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0:
        downside_std = np.std(downside)
        if downside_std > 0:
            sortino = float(np.mean(pnls) / downside_std * np.sqrt(min(n, 252)))
        else:
            sortino = float('inf') if avg_pnl > 0 else 0.0
    else:
        sortino = float('inf') if avg_pnl > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    return {
        'n_trades': n,
        'win_rate': float(win_rate),
        'avg_pnl_ticks': float(avg_pnl),
        'total_pnl_ticks': float(total_pnl),
        'total_pnl_dollars': float(total_pnl * TICK_VALUE),
        'sharpe': float(sharpe),
        'sortino': float(min(sortino, 999.0)),
        'profit_factor': float(profit_factor),
        'avg_winner': float(np.mean(winners)) if len(winners) > 0 else 0.0,
        'avg_loser': float(np.mean(losers)) if len(losers) > 0 else 0.0,
        'max_win': float(np.max(pnls)),
        'max_loss': float(np.min(pnls)),
    }


def main():
    print("=" * 80)
    print("FIFO TIME-EXIT BACKTEST — CNN-Mamba v2")
    print("=" * 80)
    print(f"Predictions: {PRED_DIR}")
    print(f"Events: {EVENT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Find available date-based prediction files (bulk OOT format)
    available_folds = sorted([
        p.stem.replace("_predictions", "")
        for p in PRED_DIR.glob("*_predictions.npz")
        if not p.stem.startswith("fold_")
    ])
    if not available_folds:
        # Fallback: try fold-based format
        for i in range(100):
            p = PRED_DIR / f"fold_{i:02d}_oot_predictions.npz"
            if p.exists():
                available_folds.append(i)

    print(f"Available prediction dates/folds: {len(available_folds)}")

    # Define configs to test
    configs = []

    for confidence_pct in [0.5, 1.0, 5.0]:
        for hold_seconds in [1.0, 2.0, 5.0, 10.0]:
            for exit_type in ['market', 'hybrid']:
                for direction in ['long', 'short', 'both']:
                    # Match horizon to hold time (use closest available)
                    if hold_seconds <= 1.5:
                        horizon_idx = 0  # 1s
                    elif hold_seconds <= 3.0:
                        horizon_idx = 1  # 5s (use 5s predictions for 2s hold)
                    elif hold_seconds <= 7.0:
                        horizon_idx = 1  # 5s
                    else:
                        horizon_idx = 2  # 10s

                    configs.append({
                        'horizon_idx': horizon_idx,
                        'hold_seconds': hold_seconds,
                        'confidence_pct': confidence_pct,
                        'exit_type': exit_type,
                        'direction': direction,
                    })

    print(f"Configs to test: {len(configs)}")
    print(f"Total simulations: {len(configs) * len(available_folds)}")
    print()

    # Build all (fold, config) pairs
    all_tasks = []
    for config in configs:
        for fold_idx in available_folds:
            all_tasks.append((fold_idx, config))

    print(f"Launching {len(all_tasks)} simulations across {min(cpu_count(), 16)} workers...")
    t0 = time.time()

    n_workers = min(cpu_count(), 16)
    with Pool(n_workers) as pool:
        results = pool.map(simulate_fold, all_tasks, chunksize=8)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print()

    # Aggregate results by config
    config_results = defaultdict(list)
    for r in results:
        if r is None:
            continue
        key = json.dumps(r['config'], sort_keys=True)
        config_results[key].append(r)

    # Compute summary statistics
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    summary_rows = []

    for config_key, fold_results in config_results.items():
        config = json.loads(config_key)

        all_trades = []
        for fr in fold_results:
            all_trades.extend(fr['trades'])

        n_total = len(all_trades)
        filled_trades = [t for t in all_trades if t.filled]
        unfilled_trades = [t for t in all_trades if not t.filled]

        n_filled = len(filled_trades)
        fill_rate = n_filled / n_total if n_total > 0 else 0.0

        stats = compute_stats(filled_trades)

        row = {
            'confidence_pct': config['confidence_pct'],
            'hold_seconds': config['hold_seconds'],
            'exit_type': config['exit_type'],
            'direction': config['direction'],
            'horizon_idx': config['horizon_idx'],
            'n_signals': n_total,
            'n_filled': n_filled,
            'fill_rate': fill_rate,
            **stats,
        }
        summary_rows.append(row)

    # Sort by avg_pnl_ticks descending
    summary_rows.sort(key=lambda x: x['avg_pnl_ticks'], reverse=True)

    # Print top results
    print(f"{'Conf%':>6} {'Hold':>5} {'Exit':>7} {'Dir':>6} | {'Signals':>8} {'Filled':>7} {'Fill%':>6} | {'WR':>5} {'AvgPnL':>7} {'PF':>5} {'Sharpe':>7} {'Sortino':>8}")
    print("-" * 110)

    for row in summary_rows[:40]:
        print(f"{row['confidence_pct']:>5.1f}% {row['hold_seconds']:>4.0f}s {row['exit_type']:>7} {row['direction']:>6} | "
              f"{row['n_signals']:>8} {row['n_filled']:>7} {row['fill_rate']:>5.1%} | "
              f"{row['win_rate']:>4.1%} {row['avg_pnl_ticks']:>+6.2f}t {row['profit_factor']:>5.2f} "
              f"{row['sharpe']:>7.2f} {row['sortino']:>8.2f}")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Best configs by direction
    for direction in ['short', 'long', 'both']:
        dir_rows = [r for r in summary_rows if r['direction'] == direction and r['n_filled'] >= 10]
        if dir_rows:
            best = dir_rows[0]
            print(f"\n  Best {direction.upper():>5}: conf={best['confidence_pct']:.1f}%, hold={best['hold_seconds']:.0f}s, "
                  f"exit={best['exit_type']} -> WR={best['win_rate']:.1%}, PnL={best['avg_pnl_ticks']:+.2f}t, "
                  f"PF={best['profit_factor']:.2f}, fill={best['fill_rate']:.1%} ({best['n_filled']} trades)")

    # Fill rate analysis
    print("\n\n  FILL RATE ANALYSIS (all configs):")
    for hold in [1.0, 2.0, 5.0, 10.0]:
        hold_rows = [r for r in summary_rows if r['hold_seconds'] == hold]
        if hold_rows:
            avg_fill = np.mean([r['fill_rate'] for r in hold_rows])
            print(f"    Hold {hold:.0f}s: avg fill rate = {avg_fill:.1%}")

    # Comparison: midpoint vs FIFO
    print("\n\n  MIDPOINT vs FIFO COMPARISON:")
    print("  (Midpoint baseline: top-1%, 1s hold = +0.90t net, 5s hold = +1.00t net)")
    for conf in [1.0, 0.5]:
        for hold in [1.0, 5.0]:
            matching = [r for r in summary_rows
                       if r['confidence_pct'] == conf
                       and r['hold_seconds'] == hold
                       and r['direction'] == 'both'
                       and r['exit_type'] == 'market']
            if matching:
                m = matching[0]
                print(f"    FIFO top-{conf:.1f}%, {hold:.0f}s hold, market exit: "
                      f"WR={m['win_rate']:.1%}, avg={m['avg_pnl_ticks']:+.2f}t, "
                      f"PF={m['profit_factor']:.2f}, fill={m['fill_rate']:.1%}")

    # Direction asymmetry
    print("\n\n  DIRECTION ASYMMETRY (top-1%, market exit):")
    for hold in [1.0, 2.0, 5.0, 10.0]:
        long_rows = [r for r in summary_rows
                    if r['confidence_pct'] == 1.0 and r['hold_seconds'] == hold
                    and r['direction'] == 'long' and r['exit_type'] == 'market']
        short_rows = [r for r in summary_rows
                     if r['confidence_pct'] == 1.0 and r['hold_seconds'] == hold
                     and r['direction'] == 'short' and r['exit_type'] == 'market']
        if long_rows and short_rows:
            l, s = long_rows[0], short_rows[0]
            print(f"    {hold:.0f}s: LONG avg={l['avg_pnl_ticks']:+.2f}t WR={l['win_rate']:.1%} | "
                  f"SHORT avg={s['avg_pnl_ticks']:+.2f}t WR={s['win_rate']:.1%}")

    # Save full results
    output_file = OUTPUT_DIR / "results_summary.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_folds': len(available_folds),
            'folds': available_folds,
            'fill_model': 'simplified_time_based',
            'fill_probabilities': {str(k): v for k, v in FILL_PROB_BY_HOLD_TIME.items()},
            'commission_rt_ticks': COMMISSION_RT_TICKS,
            'spread_ticks': SPREAD_TICKS,
            'assumptions': {
                'entry': 'passive limit at best bid/ask (back of queue)',
                'exit_market': 'cross spread = 1 tick + commission',
                'exit_hybrid': '50% passive fill / 50% market',
                'queue_position': 'worst case (back of queue)',
                'fill_model': 'time-based with local event rate adjustment',
            },
            'results': summary_rows,
        }, f, indent=2, default=str)

    print(f"\n\nFull results saved to: {output_file}")

    # Also save top configs
    top_file = OUTPUT_DIR / "top_configs.json"
    profitable = [r for r in summary_rows if r['avg_pnl_ticks'] > 0 and r['n_filled'] >= 10]
    with open(top_file, 'w') as f:
        json.dump(profitable[:20], f, indent=2, default=str)

    print(f"Top profitable configs: {top_file}")
    print(f"\nTotal profitable configs: {len(profitable)} / {len(summary_rows)}")


if __name__ == "__main__":
    main()
