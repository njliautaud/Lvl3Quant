#!/usr/bin/env python3
"""
Order Cancel Rate Signal Strategy
===================================
Track cancel-to-trade ratio at each price level over rolling 30s window.
Low cancel rate = real orders = strong support/resistance.
High cancel rate = fake liquidity = weak level.
Trade: when price approaches a low-cancel-rate level, bet on bounce.

Uses MBO JSONL data files (order-by-order Rithmic data).

Expected JSONL fields:
  - timestamp (epoch ms or ISO)
  - type: "add", "cancel", "modify", "trade"/"fill"
  - price, size, side, order_id

Usage: python cancel_rate_strategy.py [--data-dir /path/to/jsonl] [--output results.json]
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np


def parse_timestamp(ts):
    """Parse timestamp to epoch seconds (float)."""
    if isinstance(ts, (int, float)):
        if ts > 1e12:
            return ts / 1000.0
        return float(ts)
    if isinstance(ts, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass
        try:
            v = float(ts)
            return v / 1000.0 if v > 1e12 else v
        except Exception:
            pass
    return None


def load_mbo_events(filepath):
    """
    Load all MBO events from a JSONL file.
    Returns list of dicts with normalized fields: ts, type, price, size, side.
    """
    events = []
    type_map = {
        "add": "add", "ADD": "add", "A": "add", "new": "add", "NEW": "add",
        "cancel": "cancel", "CANCEL": "cancel", "C": "cancel", "delete": "cancel", "DELETE": "cancel",
        "modify": "modify", "MODIFY": "modify", "M": "modify", "replace": "modify",
        "trade": "trade", "TRADE": "trade", "T": "trade", "fill": "trade", "FILL": "trade",
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_type = rec.get("type", rec.get("msg_type", rec.get("event_type", "")))
            evt_type = type_map.get(str(raw_type), None)
            if evt_type is None:
                continue

            ts = rec.get("timestamp", rec.get("ts", rec.get("time", rec.get("epoch_ms"))))
            ts_sec = parse_timestamp(ts)
            if ts_sec is None:
                continue

            price = rec.get("price", rec.get("trade_price", rec.get("last_price")))
            if price is None:
                continue
            try:
                price = float(price)
            except (ValueError, TypeError):
                continue

            size = rec.get("size", rec.get("qty", rec.get("volume", 1)))
            try:
                size = int(size) if size else 1
            except (ValueError, TypeError):
                size = 1

            side = rec.get("side", rec.get("aggressor_side", ""))
            if isinstance(side, str):
                side = side.upper()
                if side in ("B", "BUY", "BID"):
                    side = "B"
                elif side in ("S", "SELL", "ASK", "A"):
                    side = "S"
                else:
                    side = "U"
            else:
                side = "U"

            events.append({
                "ts": ts_sec,
                "type": evt_type,
                "price": price,
                "size": size,
                "side": side,
            })

    # Sort by timestamp
    events.sort(key=lambda x: x["ts"])
    return events


def quantize_price(price, tick_size=0.25):
    """Round price to nearest tick."""
    return round(round(price / tick_size) * tick_size, 4)


def compute_cancel_rates(events, window_sec=30.0, tick_size=0.25):
    """
    Compute rolling cancel-to-event ratio at each price level.

    Returns list of snapshots:
        (ts, mid_price, {price_level: cancel_rate, ...})

    We emit a snapshot every time we see a trade (since that's when we might enter).
    """
    # Rolling window queues per price level
    # Each entry: (timestamp, event_type)
    level_events = defaultdict(deque)
    snapshots = []

    # Track current best bid/ask for mid price
    last_trade_price = None

    for evt in events:
        ts = evt["ts"]
        price = quantize_price(evt["price"], tick_size)

        # Add event to level queue
        level_events[price].append((ts, evt["type"]))

        # Expire old events
        cutoff = ts - window_sec
        for p in list(level_events.keys()):
            q = level_events[p]
            while q and q[0][0] < cutoff:
                q.popleft()
            if not q:
                del level_events[p]

        # On trade events, emit a snapshot
        if evt["type"] == "trade":
            last_trade_price = price

            # Compute cancel rate at each active level
            rates = {}
            for p, q in level_events.items():
                total = len(q)
                if total < 3:
                    continue  # Not enough data at this level
                cancels = sum(1 for _, t in q if t == "cancel")
                trades_at_level = sum(1 for _, t in q if t == "trade")
                adds = sum(1 for _, t in q if t == "add")
                # Cancel rate = cancels / (cancels + trades)
                denom = cancels + trades_at_level
                if denom > 0:
                    rates[p] = cancels / denom
                else:
                    rates[p] = 0.0

            if rates and last_trade_price is not None:
                snapshots.append((ts, last_trade_price, rates))

    return snapshots


def find_strong_levels(cancel_rates, max_cancel_rate, min_events=5, nearby_ticks=5, tick_size=0.25):
    """
    From a cancel_rates dict, find price levels with low cancel rate
    (real orders = strong support/resistance).

    Returns list of (price, cancel_rate) for strong levels near current price.
    """
    strong = []
    for price, rate in cancel_rates.items():
        if rate <= max_cancel_rate:
            strong.append((price, rate))
    return strong


def run_strategy(events, window_sec, max_cancel_rate, proximity_ticks, hold_sec,
                 tick_size=0.25, tick_value=12.50, snapshot_step=50):
    """
    Run cancel rate bounce strategy.

    Logic:
    - At each trade, compute cancel rate at all price levels (rolling window).
    - Identify "strong" levels (cancel_rate < max_cancel_rate).
    - If current price is within proximity_ticks of a strong level:
        - If strong level is BELOW current price: LONG (support bounce)
        - If strong level is ABOVE current price: SHORT (resistance bounce)
    - Hold for hold_sec seconds, measure PnL.

    snapshot_step: process every Nth snapshot to save compute (still same signals,
    just sampled less frequently).
    """
    print(f"    Computing cancel rates (window={window_sec}s)...")
    t0 = time.time()
    snapshots = compute_cancel_rates(events, window_sec=window_sec, tick_size=tick_size)
    t1 = time.time()
    print(f"    {len(snapshots):,} snapshots in {t1-t0:.1f}s")

    if len(snapshots) < 100:
        return {
            "window_sec": window_sec,
            "max_cancel_rate": max_cancel_rate,
            "proximity_ticks": proximity_ticks,
            "hold_sec": hold_sec,
            "n_snapshots": len(snapshots),
            "n_trades": 0,
            "sharpe": 0.0,
            "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
            "skipped": True,
        }

    # Build price timeline for forward return calculation
    trade_events = [(e["ts"], e["price"]) for e in events if e["type"] == "trade"]
    ts_timeline = np.array([t[0] for t in trade_events])
    px_timeline = np.array([t[1] for t in trade_events])

    trades = []
    last_entry_ts = 0  # Prevent overlapping trades

    for idx in range(0, len(snapshots), snapshot_step):
        ts, mid, cancel_rates = snapshots[idx]

        # Cooldown: don't enter if we're already in a trade
        if ts < last_entry_ts + hold_sec:
            continue

        # Find strong levels
        strong_levels = find_strong_levels(cancel_rates, max_cancel_rate)
        if not strong_levels:
            continue

        proximity = proximity_ticks * tick_size

        # Check if any strong level is near current price
        signal = None
        best_distance = float("inf")

        for level_price, cancel_rate in strong_levels:
            distance = mid - level_price  # positive = level is below

            if abs(distance) > proximity:
                continue  # Too far

            if abs(distance) < tick_size:
                continue  # At the level, no edge

            if abs(distance) < best_distance:
                best_distance = abs(distance)
                if distance > 0:
                    # Strong level is BELOW = support. Bet on bounce UP = LONG
                    signal = "LONG"
                else:
                    # Strong level is ABOVE = resistance. Bet on bounce DOWN = SHORT
                    signal = "SHORT"

        if signal is None:
            continue

        # Compute forward return
        exit_ts = ts + hold_sec
        entry_idx = np.searchsorted(ts_timeline, ts, side="left")
        exit_idx = np.searchsorted(ts_timeline, exit_ts, side="left")

        if entry_idx >= len(ts_timeline) or exit_idx >= len(ts_timeline):
            continue

        entry_price = px_timeline[entry_idx]
        exit_price = px_timeline[exit_idx]
        raw_return = (exit_price - entry_price) / tick_size

        if signal == "LONG":
            pnl = raw_return
        else:
            pnl = -raw_return

        trades.append({
            "ts": ts,
            "signal": signal,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl_ticks": float(pnl),
        })

        last_entry_ts = ts

    n_trades = len(trades)
    if n_trades < 5:
        return {
            "window_sec": window_sec,
            "max_cancel_rate": max_cancel_rate,
            "proximity_ticks": proximity_ticks,
            "hold_sec": hold_sec,
            "n_snapshots": len(snapshots),
            "n_trades": n_trades,
            "sharpe": 0.0,
            "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
            "skipped": True,
        }

    pnls = np.array([t["pnl_ticks"] for t in trades])
    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if pnls.std() > 0 else 1e-9
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252)
    win_rate = float((pnls > 0).mean())
    long_trades = sum(1 for t in trades if t["signal"] == "LONG")
    short_trades = sum(1 for t in trades if t["signal"] == "SHORT")

    return {
        "window_sec": window_sec,
        "max_cancel_rate": round(max_cancel_rate, 2),
        "proximity_ticks": proximity_ticks,
        "hold_sec": hold_sec,
        "n_snapshots": len(snapshots),
        "n_trades": n_trades,
        "sharpe": round(float(sharpe), 4),
        "total_pnl_ticks": round(total_pnl, 2),
        "total_pnl_usd": round(total_pnl * tick_value, 2),
        "win_rate": round(win_rate, 4),
        "avg_pnl_ticks": round(avg_pnl, 4),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Order Cancel Rate Signal Strategy Backtest")
    parser.add_argument("--data-dir", default="/home/jupiter/Lvl3Quant/data/",
                        help="Directory containing MBO JSONL files")
    parser.add_argument("--output", default="cancel_rate_strategy_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--tick-size", type=float, default=0.25,
                        help="Tick size (ES=0.25)")
    parser.add_argument("--tick-value", type=float, default=12.50,
                        help="Dollar value per tick (ES=$12.50)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max files to process (0=all)")
    parser.add_argument("--snapshot-step", type=int, default=50,
                        help="Process every Nth snapshot (speed vs resolution tradeoff)")
    args = parser.parse_args()

    print("=" * 60)
    print("Order Cancel Rate Signal Strategy")
    print("=" * 60)

    # Load MBO data
    pattern = os.path.join(args.data_dir, "*_rithmic.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        for alt in ["*.jsonl", "*mbo*.jsonl"]:
            files = sorted(glob.glob(os.path.join(args.data_dir, alt)))
            if files:
                break
    if not files:
        print(f"No JSONL files found in {args.data_dir}")
        sys.exit(1)

    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} JSONL files")

    # Load all events
    all_events = []
    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        events = load_mbo_events(filepath)
        n_trades = sum(1 for e in events if e["type"] == "trade")
        n_cancels = sum(1 for e in events if e["type"] == "cancel")
        n_adds = sum(1 for e in events if e["type"] == "add")
        print(f"  [{i+1}/{len(files)}] {fname}: {len(events):,} events "
              f"(trades={n_trades:,} adds={n_adds:,} cancels={n_cancels:,})")
        all_events.extend(events)

    print(f"\nTotal events loaded: {len(all_events):,}")

    if len(all_events) < 1000:
        print("Insufficient event data. Exiting.")
        sys.exit(1)

    # Sort by timestamp
    all_events.sort(key=lambda x: x["ts"])

    # Parameter sweep
    # Cancel rate computation is expensive, so we pre-compute for each window_sec
    # and then sweep threshold/proximity/hold
    window_sec_range = [15, 30, 60]
    max_cancel_rate_range = [0.3, 0.5, 0.7, 0.85]
    proximity_range = [2, 4, 8]         # ticks from strong level
    hold_sec_range = [10, 30, 60, 120]  # seconds

    total_configs = (len(window_sec_range) * len(max_cancel_rate_range)
                     * len(proximity_range) * len(hold_sec_range))
    print(f"\nSweeping {total_configs} configurations...")
    print(f"  window_sec: {window_sec_range}")
    print(f"  max_cancel_rate: {max_cancel_rate_range}")
    print(f"  proximity_ticks: {proximity_range}")
    print(f"  hold_sec: {hold_sec_range}")

    results = []
    best_sharpe = -np.inf
    best_config = None
    t0 = time.time()
    done = 0

    for window_sec in window_sec_range:
        # Pre-compute snapshots for this window (most expensive step)
        print(f"\n  Computing cancel rate snapshots for window={window_sec}s...")
        snapshots_cache = compute_cancel_rates(all_events, window_sec=window_sec,
                                                tick_size=args.tick_size)
        print(f"  {len(snapshots_cache):,} snapshots computed")

        # Build price timeline once
        trade_events = [(e["ts"], e["price"]) for e in all_events if e["type"] == "trade"]
        ts_timeline = np.array([t[0] for t in trade_events])
        px_timeline = np.array([t[1] for t in trade_events])

        for max_cancel_rate in max_cancel_rate_range:
            for proximity_ticks in proximity_range:
                for hold_sec in hold_sec_range:
                    # Run strategy using cached snapshots directly
                    # (inline the logic to avoid recomputing cancel rates)
                    trades_list = []
                    last_entry_ts = 0
                    proximity = proximity_ticks * args.tick_size

                    for idx in range(0, len(snapshots_cache), args.snapshot_step):
                        ts, mid, cancel_rates = snapshots_cache[idx]

                        if ts < last_entry_ts + hold_sec:
                            continue

                        signal = None
                        best_distance = float("inf")

                        for level_price, rate in cancel_rates.items():
                            if rate > max_cancel_rate:
                                continue
                            distance = mid - level_price
                            if abs(distance) > proximity or abs(distance) < args.tick_size:
                                continue
                            if abs(distance) < best_distance:
                                best_distance = abs(distance)
                                signal = "LONG" if distance > 0 else "SHORT"

                        if signal is None:
                            continue

                        exit_ts = ts + hold_sec
                        entry_idx = np.searchsorted(ts_timeline, ts, side="left")
                        exit_idx = np.searchsorted(ts_timeline, exit_ts, side="left")

                        if entry_idx >= len(ts_timeline) or exit_idx >= len(ts_timeline):
                            continue

                        entry_price = px_timeline[entry_idx]
                        exit_price = px_timeline[exit_idx]
                        raw_return = (exit_price - entry_price) / args.tick_size
                        pnl = raw_return if signal == "LONG" else -raw_return

                        trades_list.append(pnl)
                        last_entry_ts = ts

                    n_trades = len(trades_list)
                    if n_trades >= 5:
                        pnls = np.array(trades_list)
                        total_pnl = float(pnls.sum())
                        avg_pnl = float(pnls.mean())
                        std_pnl = float(pnls.std()) if pnls.std() > 0 else 1e-9
                        sharpe = (avg_pnl / std_pnl) * np.sqrt(252)
                        win_rate = float((pnls > 0).mean())
                    else:
                        sharpe = 0.0
                        total_pnl = 0.0
                        win_rate = 0.0
                        avg_pnl = 0.0

                    result = {
                        "window_sec": window_sec,
                        "max_cancel_rate": round(max_cancel_rate, 2),
                        "proximity_ticks": proximity_ticks,
                        "hold_sec": hold_sec,
                        "n_trades": n_trades,
                        "sharpe": round(float(sharpe), 4),
                        "total_pnl_ticks": round(total_pnl, 2),
                        "total_pnl_usd": round(total_pnl * args.tick_value, 2),
                        "win_rate": round(win_rate, 4),
                        "avg_pnl_ticks": round(avg_pnl, 4),
                        "skipped": n_trades < 5,
                    }
                    results.append(result)

                    if sharpe > best_sharpe and n_trades >= 5:
                        best_sharpe = sharpe
                        best_config = result

                    done += 1
                    if done % 20 == 0:
                        elapsed = time.time() - t0
                        print(f"    [{done}/{total_configs}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} configs in {elapsed:.1f}s")

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)
    for i, r in enumerate(results[:10]):
        if r.get("skipped"):
            continue
        print(f"  #{i+1}: Sharpe={r['sharpe']:+.3f}  PnL=${r['total_pnl_usd']:+,.0f}  "
              f"WR={r['win_rate']:.1%}  Trades={r['n_trades']:,}  "
              f"window={r['window_sec']}s cancel<={r['max_cancel_rate']} "
              f"prox={r['proximity_ticks']}t hold={r['hold_sec']}s")

    # Save
    output = {
        "strategy": "order_cancel_rate_bounce",
        "data_dir": args.data_dir,
        "n_files": len(files),
        "n_total_events": len(all_events),
        "tick_size": args.tick_size,
        "tick_value": args.tick_value,
        "snapshot_step": args.snapshot_step,
        "n_configs": len(results),
        "best_config": best_config,
        "top_20": results[:20],
        "all_results": results,
        "files_used": [os.path.basename(f) for f in files],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
