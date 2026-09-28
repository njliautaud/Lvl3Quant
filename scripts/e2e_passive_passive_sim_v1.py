#!/usr/bin/env python3
"""
End-to-End Passive Entry + Passive Exit Short Strategy Simulator (v1)

Uses FOLD-level walk-forward OOT predictions (IC~0.22) mapped to MBO events.

Trade lifecycle:
  1. CNN-Mamba v2 short signal (prediction < threshold)
  2. ENTRY: passive sell limit at ask - fill when price goes up within cancel_window
  3. EXIT: passive buy limit at bid (1 tick profit) or market exit at hold_timeout

Cost model (ES Futures, AMP/Rithmic):
  - Commission: $4.70 RT = 0.376 ticks
  - Spread: 1 tick during RTH
  - Passive exit P&L = +1.0 - 0.376 = +0.624 ticks
  - Market exit P&L = -D - 0.376 ticks (D = mid price change from entry)
"""

import numpy as np
import json
import os
import glob
import time
from itertools import product
from scipy.stats import spearmanr

# ── Constants ──────────────────────────────────────────────────────────────────
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
TICK_VALUE_USD = 12.50

PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_all_oot'
EVENT_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/e2e_passive_passive_v1'

# Fold-to-event mapping (verified by label matching)
FOLD_STRIDE = 500
FOLD_OFFSET = 997

CONFIDENCE_PCTS = [1, 2, 5, 10]
CANCEL_WINDOWS_S = [5, 10, 30]
HOLD_TIMEOUTS_S = [10, 30, 60, 120]

MIN_SIGNAL_GAP_S = 5.0

# Passive exit: price must drop this many ticks from entry mid for bid fill
# We sold at ask (mid + 0.5). Bid = mid - 0.5 = ask - 1.0.
# For exit fill, price must reach bid. Using labels (mid-to-mid moves):
# If mid drops >= 0.5 from entry, new bid is below entry, so our buy at original bid fills.
# More conservatively, require full -1.0 tick move for certainty.
# Testing both thresholds.
PASSIVE_EXIT_THRESHOLD = -1.0  # ticks (conservative: full tick drop required)


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_fold_data(fold_file):
    """Load fold predictions and map to event data."""
    if not os.path.exists(fold_file):
        return None
    try:
        fold = np.load(fold_file, allow_pickle=True)
    except Exception:
        return None
    oot_file = str(fold['oot_files'][0])
    date = os.path.basename(oot_file)[:8]

    event_file = os.path.join(EVENT_DIR, f'{date}_mbo_events.npz')
    if not os.path.exists(event_file):
        return None

    events = np.load(event_file)
    preds = fold['predictions'][:, 0]  # 1s horizon
    n_preds = len(preds)

    # Map prediction indices to event indices
    event_indices = FOLD_OFFSET + np.arange(n_preds) * FOLD_STRIDE
    n_events = len(events['timestamps'])
    valid = event_indices < n_events
    event_indices = event_indices[valid]
    preds = preds[valid]

    timestamps = events['timestamps']
    labels_1s = events['labels_1s']
    labels_5s = events['labels_5s']
    labels_10s = events['labels_10s']
    labels_30s = events['labels_30s']

    # Pre-extract labels at prediction event indices
    l1 = np.nan_to_num(labels_1s[event_indices], nan=0.0)
    l5 = np.nan_to_num(labels_5s[event_indices], nan=0.0)
    l10 = np.nan_to_num(labels_10s[event_indices], nan=0.0)
    l30 = np.nan_to_num(labels_30s[event_indices], nan=0.0)

    pred_timestamps = timestamps[event_indices]

    # Verify IC
    event_labels = labels_1s[event_indices]
    bv = ~np.isnan(event_labels)
    if bv.sum() > 100:
        ic = spearmanr(preds[bv], event_labels[bv])[0]
    else:
        ic = 0.0

    return {
        'date': date,
        'preds': preds,
        'event_indices': event_indices,
        'pred_timestamps': pred_timestamps,
        'l1': l1, 'l5': l5, 'l10': l10, 'l30': l30,
        'timestamps': timestamps,
        'labels_1s': labels_1s,
        'labels_5s': labels_5s,
        'labels_10s': labels_10s,
        'labels_30s': labels_30s,
        'n_events': n_events,
        'ic_1s': ic,
        'n_preds': len(preds),
    }


def simulate_day(day_data, pred_threshold, cancel_window_s, hold_timeout_s):
    """
    Simulate all trades for one day.

    Entry fill logic:
      - We're selling at ask. Fill happens when buyers come to our price.
      - Check price path within cancel_window using multi-horizon labels.
      - If price goes UP at any checkpoint, we get filled (buyers present).
      - For mild negative moves (< 2 ticks in 1s), intra-bar upticks likely fill us.

    Exit fill logic:
      - After entry, place buy limit at bid (1 tick below ask = entry - 1 tick).
      - Track cumulative price change from fill event using chained 1s labels.
      - If cumulative change <= PASSIVE_EXIT_THRESHOLD at any point, passive exit fills.
      - Otherwise, market exit at timeout.
    """
    preds = day_data['preds']
    event_indices = day_data['event_indices']
    pred_ts = day_data['pred_timestamps']
    l1 = day_data['l1']
    l5 = day_data['l5']
    l10 = day_data['l10']
    l30 = day_data['l30']
    timestamps = day_data['timestamps']
    labels_1s = day_data['labels_1s']
    labels_5s = day_data['labels_5s']
    labels_10s = day_data['labels_10s']
    labels_30s = day_data['labels_30s']
    n_events = day_data['n_events']

    # Find short signals
    signal_mask = preds <= pred_threshold
    signal_indices = np.where(signal_mask)[0]
    n_raw_signals = int(signal_mask.sum())

    if len(signal_indices) == 0:
        return [], n_raw_signals

    # Enforce minimum gap
    signal_ts_s = pred_ts[signal_indices].astype(np.float64) / 1e9
    keep = np.ones(len(signal_indices), dtype=bool)
    last_ts = -1e18
    for i in range(len(signal_indices)):
        if signal_ts_s[i] - last_ts >= MIN_SIGNAL_GAP_S:
            last_ts = signal_ts_s[i]
        else:
            keep[i] = False
    signal_indices = signal_indices[keep]

    if len(signal_indices) == 0:
        return [], n_raw_signals

    trades = []

    for sig_i in signal_indices:
        ev_idx = event_indices[sig_i]

        # ── ENTRY FILL ─────────────────────────────────────────────
        # Check price path within cancel_window using labels at signal event
        sig_l1 = l1[sig_i]
        sig_l5 = l5[sig_i]
        sig_l10 = l10[sig_i]
        sig_l30 = l30[sig_i]

        entry_filled = False
        fill_time_s = 0.0

        # Check 1s: price goes up or stays flat = buyers active = fill likely
        if sig_l1 >= 0.0:
            entry_filled = True
            fill_time_s = 0.5  # Fill within ~0.5s on average
        elif sig_l1 > -2.0:
            # Mild down at 1s - likely intra-second uptick
            entry_filled = True
            fill_time_s = 0.5

        if not entry_filled and cancel_window_s >= 5:
            if sig_l5 >= -1.0:
                # Even with mild 5s drop, upticks within that period are very likely
                entry_filled = True
                fill_time_s = 2.5

        if not entry_filled and cancel_window_s >= 10:
            if sig_l10 >= -2.0:
                entry_filled = True
                fill_time_s = 5.0

        if not entry_filled and cancel_window_s >= 30:
            if sig_l30 >= -3.0:
                entry_filled = True
                fill_time_s = 15.0

        if not entry_filled:
            continue

        # ── EXIT SIMULATION ────────────────────────────────────────
        # Find fill event
        t0 = timestamps[ev_idx]
        fill_t = t0 + int(fill_time_s * 1e9)
        fill_ev = min(np.searchsorted(timestamps, fill_t), n_events - 1)

        # Build price path from fill event using chained 1s labels
        # Step through at 1s intervals, accumulating price changes
        checkpoints = []
        cum_price = 0.0
        cur_ev = fill_ev

        # Use 1s-stepping for first min(hold_timeout, 30) seconds
        short_hz = min(int(hold_timeout_s), 30)
        for s in range(1, short_hz + 1):
            if cur_ev >= n_events:
                break
            l1_val = labels_1s[cur_ev]
            if not np.isnan(l1_val):
                cum_price += l1_val
            target_t = fill_t + int(s * 1e9)
            next_ev = min(np.searchsorted(timestamps, target_t), n_events - 1)
            checkpoints.append((s, cum_price))
            cur_ev = next_ev

        # For longer hold timeouts, chain 30s labels
        if hold_timeout_s > 30:
            # Get 30s label from fill event for the 30s checkpoint
            l30_fill = labels_30s[fill_ev] if fill_ev < n_events else np.nan
            if not np.isnan(l30_fill):
                # Use the chained 1s labels up to 30s, then switch to 30s labels
                cum_at_30 = cum_price  # From the 1s stepping above

                for t_target in range(60, int(hold_timeout_s) + 1, 30):
                    prev_t = t_target - 30
                    prev_ev_t = fill_t + int(prev_t * 1e9)
                    prev_ev = min(np.searchsorted(timestamps, prev_ev_t), n_events - 1)
                    l30_here = labels_30s[prev_ev] if prev_ev < n_events else np.nan
                    if not np.isnan(l30_here):
                        # cum at prev_t + l30 at prev_ev = cum at t_target
                        # Find cum at prev_t from checkpoints
                        cps_at_prev = [p for t, p in checkpoints if abs(t - prev_t) < 1]
                        if cps_at_prev:
                            cum_at_target = cps_at_prev[-1] + l30_here
                        else:
                            cum_at_target = cum_at_30 + l30_here  # fallback
                        checkpoints.append((t_target, cum_at_target))
                        cum_at_30 = cum_at_target

        if not checkpoints:
            continue

        # Find passive exit or market exit
        passive_exit = False
        exit_price_change = 0.0
        exit_time = hold_timeout_s

        for t, p in checkpoints:
            if t > hold_timeout_s:
                break
            if p <= PASSIVE_EXIT_THRESHOLD:
                passive_exit = True
                exit_price_change = p
                exit_time = t
                break

        if passive_exit:
            pnl_ticks = SPREAD_TICKS - COMMISSION_TICKS  # +0.624
        else:
            # Market exit: find last checkpoint at/before timeout
            valid_cps = [(t, p) for t, p in checkpoints if t <= hold_timeout_s]
            exit_price_change = valid_cps[-1][1] if valid_cps else 0.0
            # Short P&L = -(price change) - commission
            pnl_ticks = -exit_price_change - COMMISSION_TICKS

        trades.append({
            'exit_type': 'passive' if passive_exit else 'market',
            'exit_price_change': float(exit_price_change),
            'pnl_ticks': float(pnl_ticks),
        })

    return trades, n_raw_signals


def compute_metrics(trades_by_day, config):
    """Compute aggregate metrics."""
    all_trades = []
    daily_pnls = []

    for date in sorted(trades_by_day.keys()):
        trades = trades_by_day[date]
        day_pnl = sum(t['pnl_ticks'] for t in trades)
        daily_pnls.append(day_pnl)
        all_trades.extend(trades)

    n_trades = len(all_trades)
    if n_trades == 0:
        return None

    pnl_arr = np.array([t['pnl_ticks'] for t in all_trades])
    daily_arr = np.array(daily_pnls)

    wins = int(np.sum(pnl_arr > 0))
    win_rate = wins / n_trades

    gross_profit = float(np.sum(pnl_arr[pnl_arr > 0]))
    gross_loss = float(np.abs(np.sum(pnl_arr[pnl_arr < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    downside = daily_arr[daily_arr < 0]
    if len(downside) > 0:
        ds_std = float(np.sqrt(np.mean(downside**2)))
        sortino = float(daily_arr.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else 999.0
    else:
        sortino = 999.0 if daily_arr.mean() > 0 else 0.0

    passive_exits = sum(1 for t in all_trades if t['exit_type'] == 'passive')
    market_exits = n_trades - passive_exits

    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0

    return {
        'confidence_pct': config['confidence_pct'],
        'cancel_window_s': config['cancel_window_s'],
        'hold_timeout_s': config['hold_timeout_s'],
        'n_trades': n_trades,
        'n_days': len(daily_arr),
        'trades_per_day': round(n_trades / len(daily_arr), 1),
        'total_pnl_ticks': round(float(pnl_arr.sum()), 2),
        'total_pnl_usd': round(float(pnl_arr.sum() * TICK_VALUE_USD), 2),
        'avg_pnl_ticks': round(float(pnl_arr.mean()), 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(min(pf, 999.0), 3),
        'daily_sharpe': round(sharpe, 3),
        'daily_sortino': round(min(sortino, 999.0), 3),
        'passive_exit_rate': round(passive_exits / n_trades, 4),
        'passive_exits': passive_exits,
        'market_exits': market_exits,
        'green_days': int(np.sum(daily_arr > 0)),
        'red_days': int(np.sum(daily_arr < 0)),
        'flat_days': int(np.sum(daily_arr == 0)),
        'max_drawdown_ticks': round(max_dd, 2),
        'entry_fill_rate': round(config.get('entry_fill_rate', 0), 4),
    }


def main():
    flush_print("=" * 80)
    flush_print("E2E PASSIVE-PASSIVE SHORT STRATEGY SIMULATOR v1")
    flush_print("  Using FOLD walk-forward predictions (IC~0.22)")
    flush_print("  Fold-to-event mapping: stride=500, offset=997")
    flush_print("=" * 80)

    # Load fold files
    fold_files = sorted(glob.glob(os.path.join(PRED_DIR, 'fold_*_oot_predictions.npz')))
    flush_print(f"\nFound {len(fold_files)} fold prediction files")

    # Load and verify each fold
    flush_print("\nLoading fold data and verifying IC...")
    fold_data_list = []
    for f in fold_files:
        data = load_fold_data(f)
        if data is not None:
            fold_data_list.append(data)
            flush_print(f"  {data['date']}: {data['n_preds']} preds, IC={data['ic_1s']:.3f}")

    flush_print(f"\nLoaded {len(fold_data_list)} folds with event data")
    flush_print(f"Mean IC: {np.mean([d['ic_1s'] for d in fold_data_list]):.4f}")

    # Compute global thresholds across all fold predictions
    flush_print("\nComputing prediction thresholds...")
    all_preds = np.concatenate([d['preds'] for d in fold_data_list])
    flush_print(f"Total predictions: {len(all_preds)}")

    thresholds = {}
    for pct in CONFIDENCE_PCTS:
        thresholds[pct] = float(np.percentile(all_preds, pct))
        n = int(np.sum(all_preds <= thresholds[pct]))
        flush_print(f"  Top {pct}% short: threshold={thresholds[pct]:.3f}, "
                     f"{n} signals ({n/len(fold_data_list):.0f}/day)")
    del all_preds

    # Run simulation
    configs = list(product(CONFIDENCE_PCTS, CANCEL_WINDOWS_S, HOLD_TIMEOUTS_S))
    flush_print(f"\nRunning {len(configs)} configs x {len(fold_data_list)} days...")

    all_results = {(c, cw, ht): {} for c, cw, ht in configs}
    signal_counts = {(c, cw, ht): {'signals': 0, 'filled': 0} for c, cw, ht in configs}

    t0 = time.time()
    for day_i, day_data in enumerate(fold_data_list):
        for conf_pct, cancel_s, hold_s in configs:
            threshold = thresholds[conf_pct]
            trades, n_signals = simulate_day(day_data, threshold, cancel_s, hold_s)
            key = (conf_pct, cancel_s, hold_s)
            all_results[key][day_data['date']] = trades
            signal_counts[key]['signals'] += n_signals
            signal_counts[key]['filled'] += len(trades)

        elapsed = time.time() - t0
        flush_print(f"  Day {day_i+1}/{len(fold_data_list)}: {day_data['date']} ({elapsed:.0f}s)")

    # Compute metrics
    flush_print("\nComputing metrics...")
    results = []
    for conf_pct, cancel_s, hold_s in configs:
        key = (conf_pct, cancel_s, hold_s)
        sc = signal_counts[key]
        efr = sc['filled'] / sc['signals'] if sc['signals'] > 0 else 0

        config = {
            'confidence_pct': conf_pct,
            'cancel_window_s': cancel_s,
            'hold_timeout_s': hold_s,
            'entry_fill_rate': efr,
        }
        metrics = compute_metrics(all_results[key], config)
        if metrics is not None:
            results.append(metrics)

    results.sort(key=lambda x: x['daily_sharpe'], reverse=True)

    # Print results
    flush_print("\n" + "=" * 135)
    flush_print("RESULTS SUMMARY (sorted by Daily Sharpe)")
    flush_print("=" * 135)
    header = (f"{'Conf%':>5} {'CW':>3} {'HT':>4} | {'Trades':>6} {'T/D':>5} {'Fill%':>5} | "
              f"{'AvgPnL':>7} {'WR%':>5} {'PF':>6} | {'Sharpe':>7} {'Sortino':>8} | "
              f"{'PsxEx%':>6} {'TotPnL':>8} {'$USD':>8} | "
              f"{'G':>3} {'R':>3} {'DD':>7}")
    flush_print(header)
    flush_print("-" * 135)

    for r in results:
        sortino_s = f"{r['daily_sortino']:>8.2f}" if r['daily_sortino'] < 900 else "     inf"
        pf_s = f"{r['profit_factor']:>6.2f}" if r['profit_factor'] < 900 else "   inf"
        row = (f"{r['confidence_pct']:>5} {r['cancel_window_s']:>3} {r['hold_timeout_s']:>4} | "
               f"{r['n_trades']:>6} {r['trades_per_day']:>5.1f} {100*r['entry_fill_rate']:>4.0f}% | "
               f"{r['avg_pnl_ticks']:>+7.3f} {100*r['win_rate']:>5.1f} {pf_s} | "
               f"{r['daily_sharpe']:>7.2f} {sortino_s} | "
               f"{100*r['passive_exit_rate']:>5.1f}% {r['total_pnl_ticks']:>+8.1f} {r['total_pnl_usd']:>+8.0f} | "
               f"{r['green_days']:>3} {r['red_days']:>3} {r['max_drawdown_ticks']:>7.1f}")
        flush_print(row)

    # Top 5 detailed
    flush_print("\n" + "=" * 80)
    flush_print("TOP 5 CONFIGURATIONS - DETAILED")
    flush_print("=" * 80)
    for i, r in enumerate(results[:5]):
        sortino_v = 'inf' if r['daily_sortino'] >= 900 else f"{r['daily_sortino']:.3f}"
        pf_v = 'inf' if r['profit_factor'] >= 900 else f"{r['profit_factor']:.3f}"
        flush_print(f"\n--- #{i+1}: top{r['confidence_pct']}% | cancel={r['cancel_window_s']}s | hold={r['hold_timeout_s']}s ---")
        flush_print(f"  Trades: {r['n_trades']} ({r['trades_per_day']:.1f}/day over {r['n_days']} days)")
        flush_print(f"  Entry fill rate: {100*r['entry_fill_rate']:.1f}%")
        flush_print(f"  Passive exit rate: {100*r['passive_exit_rate']:.1f}% "
                     f"({r['passive_exits']} passive, {r['market_exits']} market)")
        flush_print(f"  Avg P&L: {r['avg_pnl_ticks']:+.4f} ticks "
                     f"({r['avg_pnl_ticks']*TICK_VALUE_USD:+.2f} USD/trade)")
        flush_print(f"  Win rate: {100*r['win_rate']:.1f}%")
        flush_print(f"  Profit factor: {pf_v}")
        flush_print(f"  Daily Sharpe: {r['daily_sharpe']:.3f}")
        flush_print(f"  Daily Sortino: {sortino_v}")
        flush_print(f"  Total P&L: {r['total_pnl_ticks']:+.1f} ticks "
                     f"({r['total_pnl_usd']:+,.0f} USD)")
        flush_print(f"  Green/Red/Flat days: {r['green_days']}/{r['red_days']}/{r['flat_days']}")
        flush_print(f"  Max drawdown: {r['max_drawdown_ticks']:.1f} ticks "
                     f"({r['max_drawdown_ticks']*TICK_VALUE_USD:.0f} USD)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'simulation': 'e2e_passive_passive_v1',
            'description': 'E2E passive entry + passive exit short strategy using fold WF predictions',
            'parameters': {
                'confidence_pcts': CONFIDENCE_PCTS,
                'cancel_windows_s': CANCEL_WINDOWS_S,
                'hold_timeouts_s': HOLD_TIMEOUTS_S,
                'commission_ticks': COMMISSION_TICKS,
                'spread_ticks': SPREAD_TICKS,
                'min_signal_gap_s': MIN_SIGNAL_GAP_S,
                'passive_exit_threshold': PASSIVE_EXIT_THRESHOLD,
                'fold_stride': FOLD_STRIDE,
                'fold_offset': FOLD_OFFSET,
            },
            'data': {
                'n_folds': len(fold_data_list),
                'mean_ic_1s': round(float(np.mean([d['ic_1s'] for d in fold_data_list])), 4),
                'pred_source': 'fold_XX_oot_predictions.npz (walk-forward)',
            },
            'results': results,
        }, f, indent=2)
    flush_print(f"\nResults saved to {output_file}")

    # Key finding
    best = results[0]
    sortino_best = 'inf' if best['daily_sortino'] >= 900 else f"{best['daily_sortino']:.2f}"
    flush_print("\n" + "=" * 80)
    flush_print("KEY FINDING")
    flush_print("=" * 80)
    flush_print(f"Best config: top {best['confidence_pct']}% confidence, "
                 f"{best['cancel_window_s']}s cancel, {best['hold_timeout_s']}s hold")
    flush_print(f"  Sharpe: {best['daily_sharpe']:.2f} | Sortino: {sortino_best}")
    flush_print(f"  {best['avg_pnl_ticks']:+.3f} ticks/trade | "
                 f"{100*best['win_rate']:.1f}% WR | PF {best['profit_factor']:.2f}")
    flush_print(f"  {best['trades_per_day']:.0f} trades/day | "
                 f"{100*best['passive_exit_rate']:.0f}% passive exits")
    flush_print(f"  Total: {best['total_pnl_ticks']:+.0f} ticks "
                 f"({best['total_pnl_usd']:+,.0f} USD) over {best['n_days']} days")


if __name__ == '__main__':
    main()
