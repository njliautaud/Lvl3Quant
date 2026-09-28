#!/usr/bin/env python3
"""
V4 Multihead — Multi-Fold FIFO Tick-Level Execution Simulator
==============================================================
Runs FIFO tick-level simulation for any v4 fold that has matching MBO trade data.
Automatically discovers fold → date mapping and runs simulation.

Usage:
    python3 v4_fifo_tick_sim_multi.py                    # all folds with MBO data
    python3 v4_fifo_tick_sim_multi.py --fold 140          # specific fold
    python3 v4_fifo_tick_sim_multi.py --fold 141 --hold 10  # custom hold
"""
import argparse
import json
import numpy as np
from pathlib import Path
import datetime
import time

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
MBO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
TRADES_DIR = ROOT / "data" / "derived" / "mid_price_cache_hc439"
OUT_DIR = ROOT / "output" / "tick_level_replay"
OUT_DIR.mkdir(exist_ok=True)

STRIDE = 2000
CONF_PERCENTILE = 80
CANCEL_WINDOW_S = 10.0
DEFAULT_HOLD_S = 5.0
EXIT_CANCEL_S = 10.0
COST_RT_TICKS = 0.376
MARKET_CROSS_TICKS = 1.0
TICK_SIZE_PTS = 0.25
TICK_VALUE_USD = 12.50


def get_rth_bounds(date_str):
    """Return RTH start/end/last_entry in nanoseconds for a given date string YYYYMMDD."""
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    # RTH: 9:30-16:00 ET = 14:30-21:00 UTC (EST+5; EDT+4)
    # March-November = EDT (UTC-4), November-March = EST (UTC-5)
    # Simple heuristic: March 8 - Nov 1 → EDT
    if (m > 3 or (m == 3 and d >= 8)) and (m < 11 or (m == 11 and d < 1)):
        utc_offset = 4  # EDT
    else:
        utc_offset = 5  # EST

    rth_start = datetime.datetime(y, m, d, 9 + utc_offset, 30, 0)
    rth_end = datetime.datetime(y, m, d, 16 + utc_offset, 0, 0)
    last_entry = datetime.datetime(y, m, d, 15 + utc_offset, 55, 0)

    return (
        int(rth_start.timestamp() * 1e9),
        int(rth_end.timestamp() * 1e9),
        int(last_entry.timestamp() * 1e9),
    )


def find_fill(trade_ts, trade_prices, signal_ts_ns, limit_price_ticks, side, cancel_ns):
    """Check if a passive limit order would be filled within cancel window."""
    start_idx = np.searchsorted(trade_ts, signal_ts_ns, side='left')
    end_ts = signal_ts_ns + cancel_ns
    end_idx = np.searchsorted(trade_ts, end_ts, side='right')

    if start_idx >= len(trade_ts):
        return None, None

    window_prices = trade_prices[start_idx:end_idx]
    window_ts = trade_ts[start_idx:end_idx]

    if len(window_prices) == 0:
        return None, None

    if side == 'short':
        fill_mask = window_prices >= limit_price_ticks
    else:
        fill_mask = window_prices <= limit_price_ticks

    if not fill_mask.any():
        return None, None

    first_fill = fill_mask.argmax()
    return window_ts[first_fill], window_prices[first_fill]


def get_price_at_time(trade_ts, trade_prices, target_ts_ns):
    """Get the most recent trade price as proxy for mid price."""
    idx = np.searchsorted(trade_ts, target_ts_ns, side='right') - 1
    if idx < 0:
        return None
    return trade_prices[idx]


def simulate_fold(fold_path, hold_s=DEFAULT_HOLD_S, conf_pct=None):
    """Run FIFO sim for a single fold. Returns results dict or None."""
    d = np.load(str(fold_path), allow_pickle=True)
    fold = int(d['fold'])
    date_str = str(d['oot_files'][0]).split('/')[-1].split('_')[0]

    # Find matching MBO and trades data
    trades_file = TRADES_DIR / f"{date_str}_trades.npz"
    mbo_file = MBO_DIR / f"{date_str}_mbo_events.npz"

    if not trades_file.exists():
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': 'no trades file'}
    if not mbo_file.exists():
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': 'no MBO events file'}

    # Load data
    preds_1s = d['preds_dir'][:, 0]
    labels_1s = d['labels_dir'][:, 0]

    mbo_data = np.load(str(mbo_file))
    event_ts = mbo_data['timestamps']

    trade_data = np.load(str(trades_file))
    trade_ts = trade_data['ts_ns']
    trade_price_raw = trade_data['price_raw']
    trade_prices = trade_price_raw / (1e9 * TICK_SIZE_PTS)

    if len(trade_ts) < 100:
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': f'only {len(trade_ts)} trades'}

    # Map predictions to timestamps — auto-detect stride from data
    n_preds = len(preds_1s)
    n_events = len(event_ts)
    detected_stride = max(1, round(n_events / n_preds))
    pred_indices = np.clip(np.arange(n_preds) * detected_stride, 0, n_events - 1)
    pred_timestamps = event_ts[pred_indices]

    # RTH bounds
    rth_start, rth_end, last_entry = get_rth_bounds(date_str)

    # Filter
    valid = ~np.isnan(labels_1s)
    rth_mask = (pred_timestamps >= rth_start) & (pred_timestamps <= last_entry)
    usable = valid & rth_mask
    usable_idx = np.where(usable)[0]

    if len(usable_idx) < 10:
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': f'only {len(usable_idx)} RTH preds'}

    # Confidence filter
    abs_preds = np.abs(preds_1s[usable_idx])
    pct = conf_pct if conf_pct is not None else CONF_PERCENTILE
    threshold = np.percentile(abs_preds, pct)
    conf_mask = abs_preds >= threshold
    signal_indices = usable_idx[conf_mask]

    # RTH trades
    rth_t_mask = (trade_ts >= rth_start) & (trade_ts <= rth_end)
    rth_t_idx = np.where(rth_t_mask)[0]
    if len(rth_t_idx) < 100:
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': f'only {len(rth_t_idx)} RTH trades'}

    rth_trade_ts = trade_ts[rth_t_idx[0]:rth_t_idx[-1]+1]
    rth_trade_prices = trade_prices[rth_t_idx[0]:rth_t_idx[-1]+1]

    cancel_ns = int(CANCEL_WINDOW_S * 1e9)
    hold_ns = int(hold_s * 1e9)
    exit_cancel_ns = int(EXIT_CANCEL_S * 1e9)

    # Simulate
    trades_list = []
    n_no_price = 0
    n_not_filled = 0
    n_passive_exit = 0
    n_market_exit = 0

    for sig_idx in signal_indices:
        pred_val = preds_1s[sig_idx]
        label_val = labels_1s[sig_idx]
        sig_ts = pred_timestamps[sig_idx]
        side = 'short' if pred_val < 0 else 'long'

        current_price = get_price_at_time(rth_trade_ts, rth_trade_prices, sig_ts)
        if current_price is None:
            n_no_price += 1
            continue

        entry_limit = current_price
        entry_fill_ts, entry_fill_price = find_fill(
            rth_trade_ts, rth_trade_prices, sig_ts, entry_limit, side, cancel_ns
        )

        if entry_fill_ts is None:
            n_not_filled += 1
            continue

        # Exit after hold
        exit_start_ts = entry_fill_ts + hold_ns
        exit_price = get_price_at_time(rth_trade_ts, rth_trade_prices, exit_start_ts)
        if exit_price is None:
            continue

        exit_side = 'long' if side == 'short' else 'short'
        exit_fill_ts, exit_fill_price = find_fill(
            rth_trade_ts, rth_trade_prices, exit_start_ts, exit_price, exit_side, exit_cancel_ns
        )

        if exit_fill_ts is not None:
            n_passive_exit += 1
            cost = COST_RT_TICKS
            actual_exit = exit_fill_price
        else:
            n_market_exit += 1
            cost = COST_RT_TICKS + MARKET_CROSS_TICKS
            forced_ts = exit_start_ts + exit_cancel_ns
            actual_exit = get_price_at_time(rth_trade_ts, rth_trade_prices, forced_ts)
            if actual_exit is None:
                actual_exit = exit_price
            exit_fill_ts = forced_ts

        if side == 'short':
            gross = entry_fill_price - actual_exit
        else:
            gross = actual_exit - entry_fill_price

        net = gross - cost
        trades_list.append({
            'side': side,
            'pred': float(pred_val),
            'label': float(label_val),
            'gross_pnl': float(gross),
            'net_pnl': float(net),
            'exit_type': 'passive' if cost == COST_RT_TICKS else 'market',
        })

    if not trades_list:
        return {'fold': fold, 'date': date_str, 'status': 'SKIP', 'reason': 'no fills'}

    # Stats
    net_pnls = np.array([t['net_pnl'] for t in trades_list])
    n_trades = len(trades_list)
    total_net = float(net_pnls.sum())
    avg_net = float(net_pnls.mean())
    wr = float((net_pnls > 0).mean())

    # Sharpe
    if net_pnls.std() > 0:
        sharpe_trade = float(net_pnls.mean() / net_pnls.std())
        sharpe_ann = float(sharpe_trade * np.sqrt(n_trades * 250))
    else:
        sharpe_trade = sharpe_ann = 0.0

    # Sortino
    down = net_pnls[net_pnls < 0]
    if len(down) > 0 and down.std() > 0:
        sortino_ann = float(net_pnls.mean() / down.std() * np.sqrt(n_trades * 250))
    else:
        sortino_ann = 0.0

    # PF
    wins = net_pnls[net_pnls > 0].sum()
    losses = abs(net_pnls[net_pnls < 0].sum())
    pf = float(wins / losses) if losses > 0 else float('inf')

    # Side breakdown
    short_net = [t['net_pnl'] for t in trades_list if t['side'] == 'short']
    long_net = [t['net_pnl'] for t in trades_list if t['side'] == 'long']

    return {
        'fold': fold,
        'date': date_str,
        'status': 'OK',
        'hold_s': hold_s,
        'n_signals': len(signal_indices),
        'n_trades': n_trades,
        'fill_rate': round(n_trades / max(1, len(signal_indices) - n_no_price) * 100, 1),
        'passive_exits': n_passive_exit,
        'market_exits': n_market_exit,
        'total_net_ticks': round(total_net, 2),
        'total_net_usd': round(total_net * TICK_VALUE_USD, 2),
        'avg_net': round(avg_net, 4),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 2),
        'sharpe_ann': round(sharpe_ann, 1),
        'sortino_ann': round(sortino_ann, 1),
        'short_n': len(short_net),
        'short_net': round(sum(short_net), 2),
        'long_n': len(long_net),
        'long_net': round(sum(long_net), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, help='Test specific fold')
    parser.add_argument('--hold', type=float, default=DEFAULT_HOLD_S, help='Hold period in seconds')
    parser.add_argument('--conf', type=int, default=None, help='Confidence percentile (0-99, default 80)')
    args = parser.parse_args()

    folds = sorted(PRED_DIR.glob("fold_*_oot_predictions.npz"))
    if args.fold:
        folds = [f for f in folds if f'fold_{args.fold:03d}' in f.name or f'fold_{args.fold}' in f.name]

    print(f"Testing {len(folds)} folds with {args.hold}s hold...")

    all_results = []
    for fp in folds:
        t0 = time.time()
        result = simulate_fold(fp, hold_s=args.hold, conf_pct=args.conf)
        elapsed = time.time() - t0

        if result['status'] == 'SKIP':
            print(f"  Fold {result['fold']} ({result['date']}): SKIP ({result['reason']})")
            continue

        status = 'NET+' if result['total_net_ticks'] > 0 else 'NET-'
        print(f"  Fold {result['fold']} ({result['date']}): {result['n_trades']} trades, "
              f"net={result['total_net_ticks']:+.1f}t, avg={result['avg_net']:+.4f}t/tr, "
              f"WR={result['wr']:.0f}%, PF={result['pf']:.2f}, Sharpe={result['sharpe_ann']:.0f} "
              f"[{status}] ({elapsed:.1f}s)")

        all_results.append(result)

    # Save
    out_path = OUT_DIR / f"v4_fifo_sim_hold{args.hold:.0f}s.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    # Summary
    if all_results:
        profitable = [r for r in all_results if r['total_net_ticks'] > 0]
        total_net = sum(r['total_net_ticks'] for r in all_results)
        total_n = sum(r['n_trades'] for r in all_results)
        avg_net = total_net / total_n if total_n > 0 else 0

        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(all_results)} days, hold={args.hold}s):")
        print(f"  {len(profitable)}/{len(all_results)} profitable")
        print(f"  Total: {total_net:+.0f}t ({total_net*TICK_VALUE_USD:+,.0f} USD)")
        print(f"  Trades: {total_n}, avg {avg_net:+.4f}t/tr")

        daily_pnl = np.array([r['total_net_ticks'] for r in all_results])
        if daily_pnl.std() > 0:
            daily_sharpe = daily_pnl.mean() / daily_pnl.std()
            print(f"  Daily Sharpe: {daily_sharpe:.2f} (ann: {daily_sharpe*np.sqrt(252):.1f})")


if __name__ == '__main__':
    main()
