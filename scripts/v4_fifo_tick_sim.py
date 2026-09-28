#!/usr/bin/env python3
"""
FIFO Tick-Level Execution Simulator for V4 Multihead Fold 140
=============================================================
Simulates passive limit order entry + passive exit against actual MBO trade data.

Data:
- V4 predictions: fold_140_oot_predictions.npz (preds_dir[:,0] = 1s directional)
- MBO trades: 20260311_trades.npz (ts_ns, price_raw in fixed-point 10^9)
- MBO events: 20260311_mbo_events.npz (timestamps for prediction alignment)

Cost model: 0.376 ticks RT (passive both sides, commission only)
"""

import json
import numpy as np
from pathlib import Path
import datetime

# ── Config ──────────────────────────────────────────────────────────────────
PRED_FILE   = Path("/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1/fold_140_oot_predictions.npz")
TRADES_FILE = Path("/home/jupiter/Lvl3Quant/data/derived/mid_price_cache_hc439/20260311_trades.npz")
MBO_FILE    = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/20260311_mbo_events.npz")
OUT_FILE    = Path("/home/jupiter/Lvl3Quant/output/tick_level_replay/v4_fold140_fifo_sim.json")

STRIDE = 2000           # events per prediction
CONF_PERCENTILE = 80    # top 20% by |pred|
CANCEL_WINDOW_S = 10.0  # cancel unfilled orders after 10s
HOLD_S = 5.0            # hold period after fill
EXIT_CANCEL_S = 10.0    # cancel exit order after 10s (then market exit)
COST_RT_TICKS = 0.376   # commission only (passive both sides)
MARKET_CROSS_TICKS = 1.0 # extra cost if forced to market-exit
TICK_SIZE_PTS = 0.25
TICK_VALUE_USD = 12.50

# RTH: 9:30-16:00 ET = 14:30-21:00 UTC on 2026-03-11
RTH_START_NS = int(datetime.datetime(2026, 3, 11, 14, 30, 0).timestamp() * 1e9)
RTH_END_NS   = int(datetime.datetime(2026, 3, 11, 21, 0, 0).timestamp() * 1e9)
# Stop entering 5 min before close to allow exits
LAST_ENTRY_NS = int(datetime.datetime(2026, 3, 11, 20, 55, 0).timestamp() * 1e9)

def load_data():
    """Load and align all data sources."""
    # Predictions
    pred_data = np.load(PRED_FILE, allow_pickle=True)
    preds_1s = pred_data['preds_dir'][:, 0]      # 1s directional predictions
    labels_1s = pred_data['labels_dir'][:, 0]     # realized returns in ticks

    # MBO events (for timestamps)
    mbo_data = np.load(MBO_FILE)
    event_ts = mbo_data['timestamps']  # nanosecond timestamps for each MBO event

    # Map predictions to timestamps via stride
    n_preds = len(preds_1s)
    pred_event_indices = np.arange(n_preds) * STRIDE
    pred_event_indices = np.clip(pred_event_indices, 0, len(event_ts) - 1)
    pred_timestamps = event_ts[pred_event_indices]

    # Trades
    trade_data = np.load(TRADES_FILE)
    trade_ts = trade_data['ts_ns']
    trade_price_raw = trade_data['price_raw']
    # Convert to ticks: price_raw / 1e9 = points, / 0.25 = ticks
    trade_price_ticks = trade_price_raw / (1e9 * TICK_SIZE_PTS)

    return preds_1s, labels_1s, pred_timestamps, trade_ts, trade_price_ticks


def find_fill(trade_ts, trade_prices, signal_ts_ns, limit_price_ticks, side, cancel_ns):
    """
    Check if a passive limit order would be filled.

    For a SHORT entry: limit sell at best ask (limit_price).
        Fill when trade >= limit_price (buyer lifts the ask).
    For a LONG entry: limit buy at best bid (limit_price).
        Fill when trade <= limit_price (seller hits the bid).

    Returns (fill_ts_ns, fill_price_ticks) or (None, None) if not filled.
    """
    # Find trades in the cancel window
    start_idx = np.searchsorted(trade_ts, signal_ts_ns, side='left')
    end_ts = signal_ts_ns + cancel_ns
    end_idx = np.searchsorted(trade_ts, end_ts, side='right')

    if start_idx >= len(trade_ts):
        return None, None

    window_ts = trade_ts[start_idx:end_idx]
    window_prices = trade_prices[start_idx:end_idx]

    if len(window_prices) == 0:
        return None, None

    if side == 'short':
        # Selling at ask: filled when a trade occurs at or above our limit
        fill_mask = window_prices >= limit_price_ticks
    else:
        # Buying at bid: filled when a trade occurs at or below our limit
        fill_mask = window_prices <= limit_price_ticks

    if not fill_mask.any():
        return None, None

    first_fill = fill_mask.argmax()
    return window_ts[first_fill], window_prices[first_fill]


def get_mid_price_at_time(trade_ts, trade_prices, target_ts_ns):
    """Get the most recent trade price as proxy for mid price."""
    idx = np.searchsorted(trade_ts, target_ts_ns, side='right') - 1
    if idx < 0:
        return None
    return trade_prices[idx]


def simulate():
    """Run the full FIFO tick-level simulation."""
    print("Loading data...")
    preds_1s, labels_1s, pred_ts, trade_ts, trade_prices = load_data()

    # Filter to RTH with valid labels
    valid_mask = ~np.isnan(labels_1s)
    rth_mask = (pred_ts >= RTH_START_NS) & (pred_ts <= LAST_ENTRY_NS)
    usable_mask = valid_mask & rth_mask

    usable_indices = np.where(usable_mask)[0]
    print(f"Usable predictions in RTH: {len(usable_indices)}")

    # Confidence filter: top 20% by |pred|
    abs_preds = np.abs(preds_1s[usable_indices])
    threshold = np.percentile(abs_preds, CONF_PERCENTILE)
    conf_mask = abs_preds >= threshold
    signal_indices = usable_indices[conf_mask]
    print(f"Signals above p{CONF_PERCENTILE} (|pred| >= {threshold:.4f}): {len(signal_indices)}")

    # Also filter RTH trades
    rth_trade_mask = (trade_ts >= RTH_START_NS) & (trade_ts <= RTH_END_NS)
    rth_trade_idx = np.where(rth_trade_mask)[0]
    print(f"RTH trades: {len(rth_trade_idx)}")
    if len(rth_trade_idx) == 0:
        print("ERROR: No RTH trades found")
        return

    rth_trade_ts = trade_ts[rth_trade_idx[0]:rth_trade_idx[-1]+1]
    rth_trade_prices = trade_prices[rth_trade_idx[0]:rth_trade_idx[-1]+1]

    cancel_ns = int(CANCEL_WINDOW_S * 1e9)
    hold_ns = int(HOLD_S * 1e9)
    exit_cancel_ns = int(EXIT_CANCEL_S * 1e9)

    # ── Simulate each signal ─────────────────────────────────────────────
    trades = []
    n_no_price = 0
    n_entry_not_filled = 0
    n_exit_passive = 0
    n_exit_market = 0

    for i, sig_idx in enumerate(signal_indices):
        pred_val = preds_1s[sig_idx]
        label_val = labels_1s[sig_idx]
        sig_ts = pred_ts[sig_idx]

        # Determine side
        side = 'short' if pred_val < 0 else 'long'

        # Get current price at signal time
        current_price = get_mid_price_at_time(rth_trade_ts, rth_trade_prices, sig_ts)
        if current_price is None:
            n_no_price += 1
            continue

        # Entry limit price: passive at best bid/ask
        # ES book is 1 tick wide during RTH
        # For short: sell at ask = current_price + 0.5 ticks (half-tick above mid)
        #   But since we use last trade as proxy, and ES is 1 tick wide,
        #   we place at current trade price (which is at bid or ask already)
        # Simplification: entry at current_price (passive, queue at the level)
        if side == 'short':
            entry_limit = current_price  # selling at current level
        else:
            entry_limit = current_price  # buying at current level

        # Try to fill entry
        entry_fill_ts, entry_fill_price = find_fill(
            rth_trade_ts, rth_trade_prices, sig_ts, entry_limit, side, cancel_ns
        )

        if entry_fill_ts is None:
            n_entry_not_filled += 1
            continue

        # Entry filled. Now simulate exit after hold period.
        exit_start_ts = entry_fill_ts + hold_ns

        # Get price at exit time
        exit_price_at_hold = get_mid_price_at_time(rth_trade_ts, rth_trade_prices, exit_start_ts)
        if exit_price_at_hold is None:
            # Past end of data
            continue

        # Exit: passive at exit price level
        exit_side = 'long' if side == 'short' else 'short'  # opposite
        exit_limit = exit_price_at_hold

        exit_fill_ts, exit_fill_price = find_fill(
            rth_trade_ts, rth_trade_prices, exit_start_ts, exit_limit, exit_side, exit_cancel_ns
        )

        if exit_fill_ts is not None:
            # Passive exit
            n_exit_passive += 1
            exit_cost = COST_RT_TICKS  # passive both sides
            actual_exit_price = exit_fill_price
        else:
            # Market exit (forced)
            n_exit_market += 1
            exit_cost = COST_RT_TICKS + MARKET_CROSS_TICKS  # passive entry + market exit
            # Exit at worst price in the window
            forced_exit_ts = exit_start_ts + exit_cancel_ns
            actual_exit_price = get_mid_price_at_time(rth_trade_ts, rth_trade_prices, forced_exit_ts)
            if actual_exit_price is None:
                actual_exit_price = exit_price_at_hold
            exit_fill_ts = forced_exit_ts

        # Calculate PnL in ticks
        if side == 'short':
            gross_pnl = entry_fill_price - actual_exit_price
        else:
            gross_pnl = actual_exit_price - entry_fill_price

        net_pnl = gross_pnl - exit_cost

        trades.append({
            'signal_idx': int(sig_idx),
            'pred': float(pred_val),
            'label': float(label_val),
            'side': side,
            'entry_ts_ns': int(entry_fill_ts),
            'entry_price_ticks': float(entry_fill_price),
            'exit_ts_ns': int(exit_fill_ts) if exit_fill_ts else None,
            'exit_price_ticks': float(actual_exit_price),
            'gross_pnl_ticks': float(gross_pnl),
            'net_pnl_ticks': float(net_pnl),
            'exit_type': 'passive' if exit_fill_ts is not None and n_exit_market == 0 or (exit_fill_ts is not None and trades and trades[-1].get('exit_type') != 'market') else 'market',
            'cost_ticks': float(exit_cost),
        })
        # Fix exit_type properly
        trades[-1]['exit_type'] = 'passive' if exit_fill_ts is not None and exit_cost == COST_RT_TICKS else 'market'

    # ── Compute Statistics ───────────────────────────────────────────────
    if not trades:
        print("No trades filled!")
        return

    net_pnls = np.array([t['net_pnl_ticks'] for t in trades])
    gross_pnls = np.array([t['gross_pnl_ticks'] for t in trades])

    total_net = net_pnls.sum()
    total_gross = gross_pnls.sum()
    avg_net = net_pnls.mean()
    avg_gross = gross_pnls.mean()
    win_rate = (net_pnls > 0).sum() / len(net_pnls)

    # Sharpe (annualized from per-trade)
    if net_pnls.std() > 0:
        sharpe_per_trade = net_pnls.mean() / net_pnls.std()
        # Rough annualization: ~250 trading days, assume similar trade count per day
        trades_per_year = len(trades) * 250
        sharpe_annual = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_per_trade = 0
        sharpe_annual = 0

    # Sortino
    downside = net_pnls[net_pnls < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino_per_trade = net_pnls.mean() / downside.std()
        sortino_annual = sortino_per_trade * np.sqrt(trades_per_year)
    else:
        sortino_per_trade = 0
        sortino_annual = 0

    # Profit Factor
    gross_wins = net_pnls[net_pnls > 0].sum()
    gross_losses = abs(net_pnls[net_pnls < 0].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Side breakdown
    short_trades = [t for t in trades if t['side'] == 'short']
    long_trades = [t for t in trades if t['side'] == 'long']
    short_pnl = sum(t['net_pnl_ticks'] for t in short_trades) if short_trades else 0
    long_pnl = sum(t['net_pnl_ticks'] for t in long_trades) if long_trades else 0
    short_wr = sum(1 for t in short_trades if t['net_pnl_ticks'] > 0) / len(short_trades) if short_trades else 0
    long_wr = sum(1 for t in long_trades if t['net_pnl_ticks'] > 0) / len(long_trades) if long_trades else 0

    # Exit type breakdown
    passive_exits = sum(1 for t in trades if t['exit_type'] == 'passive')
    market_exits = sum(1 for t in trades if t['exit_type'] == 'market')

    # ── Print Results ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("V4 MULTIHEAD FOLD 140 — FIFO TICK-LEVEL SIMULATION")
    print("="*70)
    print(f"Date: 2026-03-11 (RTH 9:30-16:00 ET)")
    print(f"Model: V4 multihead pressure v1, fold 140")
    print(f"Signal threshold: top {100-CONF_PERCENTILE}% by |pred| (>= {threshold:.4f})")
    print(f"Hold period: {HOLD_S}s, Cancel window: {CANCEL_WINDOW_S}s")
    print(f"Cost model: {COST_RT_TICKS} ticks RT (passive), +{MARKET_CROSS_TICKS} tick if market exit")
    print("-"*70)
    print(f"Signals generated:     {len(signal_indices)}")
    print(f"  No price available:  {n_no_price}")
    print(f"  Entry not filled:    {n_entry_not_filled}")
    print(f"  Trades filled:       {len(trades)}")
    print(f"    Passive exit:      {passive_exits}")
    print(f"    Market exit:       {market_exits}")
    print(f"  Entry fill rate:     {len(trades)/(len(signal_indices)-n_no_price)*100:.1f}%")
    print("-"*70)
    print(f"Total gross PnL:       {total_gross:+.2f} ticks ({total_gross*TICK_VALUE_USD:+.2f} USD)")
    print(f"Total net PnL:         {total_net:+.2f} ticks ({total_net*TICK_VALUE_USD:+.2f} USD)")
    print(f"Avg gross PnL/trade:   {avg_gross:+.4f} ticks")
    print(f"Avg net PnL/trade:     {avg_net:+.4f} ticks")
    print(f"Win rate:              {win_rate*100:.1f}%")
    print(f"Profit factor:         {pf:.2f}")
    print(f"Sharpe (per-trade):    {sharpe_per_trade:.4f}")
    print(f"Sharpe (annualized):   {sharpe_annual:.2f}")
    print(f"Sortino (annualized):  {sortino_annual:.2f}")
    print("-"*70)
    print(f"SHORT trades: {len(short_trades)}, net PnL: {short_pnl:+.2f} ticks, WR: {short_wr*100:.1f}%")
    print(f"LONG  trades: {len(long_trades)}, net PnL: {long_pnl:+.2f} ticks, WR: {long_wr*100:.1f}%")
    print("="*70)

    # ── Label-based sanity check ─────────────────────────────────────────
    # Compare simulated PnL to label-implied PnL
    label_pnls = []
    for t in trades:
        # Label is realized 1s return in ticks, signed
        # For short: profit if price goes down (negative label)
        # For long: profit if price goes up (positive label)
        if t['side'] == 'short':
            label_pnls.append(-t['label'] - COST_RT_TICKS)
        else:
            label_pnls.append(t['label'] - COST_RT_TICKS)
    label_pnls = np.array(label_pnls)
    print(f"\nLabel-implied net PnL: {label_pnls.sum():+.2f} ticks (sanity check)")
    print(f"Label-implied avg:     {label_pnls.mean():+.4f} ticks/trade")
    print(f"Label-implied WR:      {(label_pnls>0).sum()/len(label_pnls)*100:.1f}%")

    # ── Save Results ─────────────────────────────────────────────────────
    results = {
        'metadata': {
            'date': '2026-03-11',
            'model': 'v4_multihead_pressure_v1',
            'fold': 140,
            'pred_file': str(PRED_FILE),
            'trades_file': str(TRADES_FILE),
            'mbo_file': str(MBO_FILE),
            'conf_percentile': CONF_PERCENTILE,
            'conf_threshold': float(threshold),
            'hold_seconds': HOLD_S,
            'cancel_window_seconds': CANCEL_WINDOW_S,
            'cost_rt_ticks': COST_RT_TICKS,
            'market_cross_ticks': MARKET_CROSS_TICKS,
        },
        'summary': {
            'signals_generated': int(len(signal_indices)),
            'trades_filled': len(trades),
            'entry_fill_rate_pct': float(round(len(trades)/(len(signal_indices)-n_no_price)*100, 1)),
            'entry_not_filled': int(n_entry_not_filled),
            'passive_exits': int(passive_exits),
            'market_exits': int(market_exits),
            'total_gross_pnl_ticks': float(round(total_gross, 4)),
            'total_net_pnl_ticks': float(round(total_net, 4)),
            'total_net_pnl_usd': float(round(total_net * TICK_VALUE_USD, 2)),
            'avg_gross_pnl_per_trade': float(round(avg_gross, 4)),
            'avg_net_pnl_per_trade': float(round(avg_net, 4)),
            'win_rate_pct': float(round(win_rate * 100, 1)),
            'profit_factor': float(round(pf, 2)),
            'sharpe_per_trade': float(round(sharpe_per_trade, 4)),
            'sharpe_annualized': float(round(sharpe_annual, 2)),
            'sortino_annualized': float(round(sortino_annual, 2)),
            'short_trades': len(short_trades),
            'short_net_pnl_ticks': float(round(short_pnl, 4)),
            'short_win_rate_pct': float(round(short_wr * 100, 1)),
            'long_trades': len(long_trades),
            'long_net_pnl_ticks': float(round(long_pnl, 4)),
            'long_win_rate_pct': float(round(long_wr * 100, 1)),
            'label_implied_net_pnl_ticks': float(round(label_pnls.sum(), 4)),
            'label_implied_avg_pnl': float(round(label_pnls.mean(), 4)),
            'label_implied_wr_pct': float(round((label_pnls>0).sum()/len(label_pnls)*100, 1)),
        },
        'trades': trades,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == '__main__':
    simulate()
