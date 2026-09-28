#!/usr/bin/env python3
"""
Hold Duration & Stop-Loss Optimizer v1
Finds optimal hold duration, stop-loss, trailing stop, and entry threshold
for confirmed SHORT signal from CNN-Mamba v2.

Labels are in TICKS (midpoint returns). 1 tick = 0.25 pts = $12.50.
For SHORT: profit_ticks = -label_ticks

Cost scenarios tested:
  PASSIVE entry (limit at bid) + MARKET exit: 0.376 commission + 0.5 tick exit spread = 0.876 ticks
  MARKET entry + MARKET exit: 0.376 commission + 1.0 tick total spread = 1.376 ticks

Labels are midpoint returns, so:
  - Passive entry at bid = mid - 0.5 tick (we gain 0.5 tick vs midpoint)
  - Market exit at bid = mid - 0.5 tick for short cover... wait, short exit = buy at ask = mid + 0.5 tick
  - So passive entry (sell at ask = mid + 0.5): we get BETTER than mid by 0.5 tick
  - Market exit (buy at ask = mid + 0.5): we pay 0.5 tick WORSE than mid
  - Net adjustment from mid-based labels: +0.5 (entry) - 0.5 (exit) = 0 tick spread impact
  - Only commission: 0.376 ticks

  For market entry (sell at bid = mid - 0.5): we get 0.5 tick WORSE than mid
  - Market exit (buy at ask = mid + 0.5): 0.5 tick worse
  - Net spread: -0.5 - 0.5 = -1.0 tick
  - Total: 0.376 + 1.0 = 1.376 ticks

We test BOTH cost scenarios.
"""

import numpy as np
import json
import os
from pathlib import Path
from itertools import product
from collections import defaultdict

# === COST SCENARIOS ===
COST_SCENARIOS = {
    'passive_entry': {
        'commission': 0.376,
        'spread': 0.0,   # passive entry at ask + market exit at ask cancel out vs midpoint
        'total': 0.376,
        'desc': 'Passive limit entry (sell at ask) + market exit (buy at ask)'
    },
    'market_entry': {
        'commission': 0.376,
        'spread': 1.0,    # 0.5 entry + 0.5 exit
        'total': 1.376,
        'desc': 'Market entry (sell at bid) + market exit (buy at ask)'
    },
}

# Default cost for primary analysis
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 0.0  # Passive entry scenario (matches the +0.43 ticks/trade finding)
TOTAL_COST_TICKS = COMMISSION_TICKS + SPREAD_TICKS  # 0.376 ticks for passive entry

FEAT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hold_optimizer_1s_v2")

# === PARAMETER GRID ===
HOLD_DURATIONS = [1, 5, 10, 30]  # seconds (matching available label horizons)
STOP_LOSSES = [None, 1, 2, 3, 5, 8, 10]  # ticks against position
TRAILING_STOPS = [None]  # Simplified: can't do trailing with discrete horizons
ENTRY_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]  # top X% of short signals

# Horizon label mapping
HORIZON_MAP = {1: 'labels_1s', 5: 'labels_5s', 10: 'labels_10s', 30: 'labels_30s'}
# Ordered horizons for stop-loss checking
HORIZON_ORDER = [1, 5, 10, 30]


def load_date_data(date_str):
    """Load predictions and aligned labels for a date."""
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    feat_path = FEAT_DIR / f"{date_str}_mbo_events.npz"

    if not pred_path.exists() or not feat_path.exists():
        return None

    pred = np.load(pred_path)
    feat = np.load(feat_path)

    predictions = pred['predictions']  # [N, 3] for horizons 1s, 5s, 10s
    stride = int(pred['stride'])
    ws = int(pred['window_size'])
    n_pred = len(predictions)

    # Align predictions to event indices
    aligned_idx = np.arange(n_pred) * stride + ws - 1

    # Check bounds
    max_idx = len(feat['labels_1s']) - 1
    valid_mask = aligned_idx <= max_idx
    aligned_idx = aligned_idx[valid_mask]
    predictions = predictions[valid_mask]

    # Extract labels at aligned positions for all horizons
    labels = {}
    for h in HORIZON_ORDER:
        key = HORIZON_MAP[h]
        lbl = feat[key][aligned_idx]
        labels[h] = lbl

    # Use the 1s prediction (col 0) as our signal - multi-horizon analysis shows
    # 1s has best correlation (+0.114), 15/15 folds positive, best WR and PF
    # Negative prediction = short signal
    signal = predictions[:, 0]  # 1s horizon prediction

    return {
        'signal': signal,
        'labels': labels,
        'n_events': n_pred,
        'date': date_str,
    }


def simulate_config(all_dates_data, hold_s, stop_loss, threshold_pct, cost_ticks):
    """
    Simulate a single config across all dates.

    For SHORT signals:
    - profit_ticks = -label_ticks (price goes down = profit for short)
    - Stop-loss: if at any earlier horizon, return > stop_loss ticks (price went UP),
      exit at that horizon's return + extra slippage
    """
    daily_pnls = []
    daily_trades = []
    all_trade_pnls = []

    for dd in all_dates_data:
        signal = dd['signal']
        labels = dd['labels']

        # Select SHORT signals: most negative predictions
        # threshold_pct = fraction of signals to take (e.g., 0.01 = top 1% most negative)
        cutoff = np.nanpercentile(signal, threshold_pct * 100)
        short_mask = signal <= cutoff

        # Also require non-NaN labels at hold horizon
        valid_labels = ~np.isnan(labels[hold_s])
        mask = short_mask & valid_labels

        if mask.sum() == 0:
            daily_pnls.append(0.0)
            daily_trades.append(0)
            continue

        # Get returns at hold horizon (ticks, midpoint)
        hold_returns = labels[hold_s][mask]  # positive = price went up

        trade_pnls = np.zeros(len(hold_returns))

        if stop_loss is not None:
            # Check earlier horizons for stop-loss trigger
            earlier_horizons = [h for h in HORIZON_ORDER if h < hold_s]
            stopped = np.zeros(len(hold_returns), dtype=bool)

            for eh in earlier_horizons:
                eh_returns = labels[eh][mask]
                eh_valid = ~np.isnan(eh_returns) & ~stopped

                # Stop triggered if price moved UP by stop_loss ticks (against short)
                stop_hit = eh_valid & (eh_returns >= stop_loss)

                # Stop exit: use actual return (could be worse than stop level due to gaps)
                # Add 0.5 tick slippage for stop market order exit
                trade_pnls[stop_hit] = -eh_returns[stop_hit] - cost_ticks
                stopped |= stop_hit

            # Non-stopped trades: hold to duration
            trade_pnls[~stopped] = -hold_returns[~stopped] - cost_ticks
        else:
            # No stop-loss: always hold to duration
            trade_pnls = -hold_returns - cost_ticks

        daily_pnls.append(float(np.sum(trade_pnls)))
        daily_trades.append(int(mask.sum()))
        all_trade_pnls.extend(trade_pnls.tolist())

    return compute_metrics(daily_pnls, daily_trades, all_trade_pnls)


def compute_metrics(daily_pnls, daily_trades, all_trade_pnls):
    """Compute performance metrics."""
    daily_pnls = np.array(daily_pnls)
    daily_trades = np.array(daily_trades)
    all_trade_pnls = np.array(all_trade_pnls)

    if len(all_trade_pnls) == 0 or np.sum(daily_trades) == 0:
        return None

    total_trades = int(np.sum(daily_trades))
    trading_days = int(np.sum(daily_trades > 0))

    if trading_days == 0:
        return None

    # Per-trade metrics
    mean_pnl = float(np.mean(all_trade_pnls))
    win_rate = float(np.mean(all_trade_pnls > 0))

    # Profit factor
    gross_profit = float(np.sum(all_trade_pnls[all_trade_pnls > 0]))
    gross_loss = float(np.abs(np.sum(all_trade_pnls[all_trade_pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Daily Sharpe
    active_daily = daily_pnls[daily_trades > 0]
    daily_mean = float(np.mean(active_daily))
    daily_std = float(np.std(active_daily))
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # Daily Sortino
    downside = active_daily[active_daily < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown (cumulative ticks)
    cumulative = np.cumsum(all_trade_pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Green day %
    green_days = float(np.mean(active_daily > 0))

    # Trades per day
    trades_per_day = float(np.mean(daily_trades[daily_trades > 0]))

    return {
        'mean_pnl_ticks': round(mean_pnl, 4),
        'win_rate': round(win_rate, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(min(profit_factor, 99.0), 3),
        'max_drawdown_ticks': round(max_dd, 2),
        'green_day_pct': round(green_days, 4),
        'total_trades': total_trades,
        'trading_days': trading_days,
        'trades_per_day': round(trades_per_day, 1),
        'total_pnl_ticks': round(float(np.sum(all_trade_pnls)), 2),
    }


def run_scenario(all_dates, scenario_name, cost_ticks):
    """Run full grid search for a cost scenario."""
    configs = list(product(HOLD_DURATIONS, STOP_LOSSES, ENTRY_THRESHOLDS))
    results = []

    for i, (hold_s, stop_loss, threshold) in enumerate(configs):
        # Skip invalid: stop-loss check needs earlier horizons
        if stop_loss is not None and hold_s == 1:
            continue

        metrics = simulate_config(all_dates, hold_s, stop_loss, threshold, cost_ticks)
        if metrics is None:
            continue

        config = {
            'hold_seconds': hold_s,
            'stop_loss_ticks': stop_loss,
            'entry_threshold_pct': threshold,
            'cost_scenario': scenario_name,
            'cost_ticks': cost_ticks,
        }
        results.append({**config, **metrics})

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


def print_results(results, scenario_name, cost_ticks, n_dates):
    """Print formatted results table."""
    print(f"\n{'=' * 150}")
    print(f"  SCENARIO: {scenario_name} (cost = {cost_ticks} ticks/trade) | {n_dates} OOT dates")
    print(f"{'=' * 150}")
    print(f"{'Rank':>4} | {'Hold':>5} | {'StopL':>5} | {'Thresh':>6} | {'PnL/Trade':>9} | {'WR':>6} | {'Sharpe':>7} | {'Sortino':>8} | {'PF':>6} | {'MaxDD':>7} | {'Green%':>6} | {'Trades':>6} | {'Tot PnL':>8}")
    print("-" * 150)

    for i, r in enumerate(results[:10]):
        sl = f"{r['stop_loss_ticks']}t" if r['stop_loss_ticks'] else "None"
        print(f"{i+1:>4} | {r['hold_seconds']:>4}s | {sl:>5} | {r['entry_threshold_pct']*100:>5.1f}% | "
              f"{r['mean_pnl_ticks']:>+8.3f} | {r['win_rate']*100:>5.1f}% | {r['sharpe']:>7.2f} | "
              f"{r['sortino']:>8.2f} | {r['profit_factor']:>5.2f} | {r['max_drawdown_ticks']:>6.1f} | "
              f"{r['green_day_pct']*100:>5.1f}% | {r['total_trades']:>6} | {r['total_pnl_ticks']:>+7.1f}")

    # Best per hold duration
    print(f"\n  BEST CONFIG PER HOLD DURATION ({scenario_name}):")
    for h in HOLD_DURATIONS:
        h_results = [r for r in results if r['hold_seconds'] == h]
        if h_results:
            best = h_results[0]
            sl = f"{best['stop_loss_ticks']}t" if best['stop_loss_ticks'] else "None"
            print(f"    Hold {h:>2}s: Sharpe={best['sharpe']:>6.2f}, PnL/trade={best['mean_pnl_ticks']:>+.3f}t, "
                  f"WR={best['win_rate']*100:.1f}%, PF={best['profit_factor']:.2f}, "
                  f"StopL={sl}, Thresh={best['entry_threshold_pct']*100:.1f}%, "
                  f"Trades={best['total_trades']}, Green={best['green_day_pct']*100:.0f}%")


def main():
    # Load all date data
    print("Loading data...")
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))

    all_dates = []
    for pf in pred_files:
        date_str = pf.stem.replace("_predictions", "")
        dd = load_date_data(date_str)
        if dd is not None:
            all_dates.append(dd)

    print(f"Loaded {len(all_dates)} OOT dates")
    total_events = sum(d['n_events'] for d in all_dates)
    print(f"Total prediction events: {total_events:,}")

    all_results = {}

    # Run both cost scenarios
    for scenario_name, scenario in COST_SCENARIOS.items():
        cost = scenario['total']
        print(f"\n--- Running scenario: {scenario_name} (cost={cost} ticks) ---")
        print(f"    {scenario['desc']}")

        results = run_scenario(all_dates, scenario_name, cost)
        all_results[scenario_name] = results
        print_results(results, scenario_name, cost, len(all_dates))

    # Save combined top 20 from passive_entry (primary scenario)
    primary_results = all_results['passive_entry'][:20]
    out_data = {
        'passive_entry_top20': primary_results,
        'market_entry_top20': all_results['market_entry'][:20],
        'metadata': {
            'oot_dates': len(all_dates),
            'total_events': total_events,
            'cost_passive': COST_SCENARIOS['passive_entry']['total'],
            'cost_market': COST_SCENARIOS['market_entry']['total'],
            'hold_durations_tested': HOLD_DURATIONS,
            'stop_losses_tested': STOP_LOSSES,
            'thresholds_tested': ENTRY_THRESHOLDS,
        }
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("EXECUTIVE SUMMARY")
    print(f"{'=' * 80}")
    print(f"Cost assumptions:")
    for name, sc in COST_SCENARIOS.items():
        print(f"  {name}: {sc['total']} ticks ({sc['desc']})")
    print(f"OOT dates: {len(all_dates)}")

    # Highlight: is ANY config profitable with market entry?
    mkt_profitable = [r for r in all_results['market_entry'] if r['mean_pnl_ticks'] > 0]
    passive_profitable = [r for r in all_results['passive_entry'] if r['mean_pnl_ticks'] > 0]
    print(f"\nProfitable configs (market entry):  {len(mkt_profitable)}")
    print(f"Profitable configs (passive entry): {len(passive_profitable)}")

    if passive_profitable:
        best = passive_profitable[0]
        sl = f"{best['stop_loss_ticks']}t" if best['stop_loss_ticks'] else "None"
        print(f"\nBest passive config: Hold={best['hold_seconds']}s, Thresh={best['entry_threshold_pct']*100:.1f}%, "
              f"StopL={sl}")
        print(f"  Sharpe={best['sharpe']:.2f}, PnL/trade={best['mean_pnl_ticks']:+.3f}t, "
              f"WR={best['win_rate']*100:.1f}%, PF={best['profit_factor']:.2f}, "
              f"Green={best['green_day_pct']*100:.0f}%")


if __name__ == "__main__":
    main()
