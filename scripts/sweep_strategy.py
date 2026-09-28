#!/usr/bin/env python3
"""
Trade Sweep Detection Strategy
===============================
Detects when multiple price levels get hit in rapid succession (market order sweep).
Signal: 3+ levels swept in <500ms = momentum signal.
Trade in direction of sweep, hold 10-60s.

Uses MBO JSONL data files (order-by-order Rithmic data).

Each JSONL line is expected to have fields like:
  - timestamp (epoch ms or ISO)
  - type: "trade", "add", "modify", "cancel"
  - price, size, side ("B"/"S"), order_id
  - (exact schema may vary — script handles common formats)

Usage: python sweep_strategy.py [--data-dir /path/to/jsonl] [--output results.json]
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np


def parse_timestamp(ts):
    """Parse timestamp to epoch seconds (float)."""
    if isinstance(ts, (int, float)):
        # If > 1e12, it's milliseconds
        if ts > 1e12:
            return ts / 1000.0
        return float(ts)
    if isinstance(ts, str):
        # Try ISO format
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass
        # Try epoch string
        try:
            v = float(ts)
            return v / 1000.0 if v > 1e12 else v
        except Exception:
            pass
    return None


def load_mbo_trades(filepath):
    """
    Load trade events from an MBO JSONL file.
    Returns list of (timestamp_sec, price, size, side) tuples.
    """
    trades = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Detect record type — handle multiple schema variants
            rec_type = rec.get("type", rec.get("msg_type", rec.get("event_type", "")))
            if rec_type not in ("trade", "fill", "Trade", "TRADE", "T"):
                # Also check if it looks like a trade by having trade-specific fields
                if "trade_price" not in rec and "last_price" not in rec:
                    continue

            ts = rec.get("timestamp", rec.get("ts", rec.get("time", rec.get("epoch_ms"))))
            ts_sec = parse_timestamp(ts)
            if ts_sec is None:
                continue

            price = rec.get("price", rec.get("trade_price", rec.get("last_price")))
            size = rec.get("size", rec.get("qty", rec.get("volume", rec.get("trade_size", 1))))
            side = rec.get("side", rec.get("aggressor_side", rec.get("direction", "")))

            if price is None:
                continue

            try:
                price = float(price)
                size = int(size) if size else 1
            except (ValueError, TypeError):
                continue

            # Normalize side
            if isinstance(side, str):
                side = side.upper()
                if side in ("B", "BUY", "BID"):
                    side = "B"
                elif side in ("S", "SELL", "ASK", "A"):
                    side = "S"
                else:
                    side = "U"  # unknown
            else:
                side = "U"

            trades.append((ts_sec, price, size, side))

    return trades


def detect_sweeps(trades, n_levels_min=3, time_window_sec=0.5, tick_size=0.25):
    """
    Detect sweep events: n_levels_min+ distinct price levels hit within time_window_sec.

    Returns list of sweep events:
        (timestamp, direction, n_levels, total_size, price_start, price_end)
    """
    if len(trades) < n_levels_min:
        return []

    trades_arr = np.array(trades, dtype=[
        ("ts", "f8"), ("price", "f8"), ("size", "i4"), ("side", "U1")
    ])
    # Sort by timestamp
    trades_arr.sort(order="ts")

    sweeps = []
    i = 0
    n = len(trades_arr)

    while i < n - n_levels_min + 1:
        # Window from trade i
        t_start = trades_arr["ts"][i]
        t_end = t_start + time_window_sec

        # Collect all trades in window
        j = i
        while j < n and trades_arr["ts"][j] <= t_end:
            j += 1

        window = trades_arr[i:j]
        if len(window) < n_levels_min:
            i += 1
            continue

        # Count distinct price levels
        unique_prices = np.unique(window["price"])
        n_levels = len(unique_prices)

        if n_levels >= n_levels_min:
            prices = window["price"]
            sizes = window["size"]
            price_start = prices[0]
            price_end = prices[-1]
            total_size = int(sizes.sum())

            # Direction: if prices are generally increasing, it's a buy sweep
            if price_end > price_start:
                direction = "UP"
            elif price_end < price_start:
                direction = "DOWN"
            else:
                # Use dominant side
                buy_vol = sizes[window["side"] == "B"].sum()
                sell_vol = sizes[window["side"] == "S"].sum()
                direction = "UP" if buy_vol > sell_vol else "DOWN"

            sweeps.append({
                "ts": float(t_start),
                "direction": direction,
                "n_levels": int(n_levels),
                "total_size": total_size,
                "price_start": float(price_start),
                "price_end": float(price_end),
            })

            # Skip past this sweep window to avoid double-counting
            i = j
        else:
            i += 1

    return sweeps


def compute_forward_returns(sweeps, trades, hold_sec, tick_size=0.25):
    """
    For each sweep event, compute the forward return over hold_sec seconds.
    Uses the trade stream to find the price at entry and exit.

    Returns array of PnL in ticks per sweep.
    """
    if not sweeps or len(trades) < 10:
        return np.array([])

    # Build price timeline: (ts, price)
    ts_arr = np.array([t[0] for t in trades])
    px_arr = np.array([t[1] for t in trades])

    pnls = []
    for sweep in sweeps:
        entry_ts = sweep["ts"]
        exit_ts = entry_ts + hold_sec

        # Entry price: first trade after sweep timestamp
        entry_idx = np.searchsorted(ts_arr, entry_ts, side="left")
        if entry_idx >= len(ts_arr):
            continue
        entry_price = px_arr[entry_idx]

        # Exit price: first trade after exit_ts
        exit_idx = np.searchsorted(ts_arr, exit_ts, side="left")
        if exit_idx >= len(ts_arr):
            continue
        exit_price = px_arr[exit_idx]

        # PnL in ticks
        raw_return = (exit_price - entry_price) / tick_size
        if sweep["direction"] == "UP":
            pnl = raw_return  # Long
        else:
            pnl = -raw_return  # Short

        pnls.append(pnl)

    return np.array(pnls)


def run_strategy(trades, n_levels_min, time_window_sec, hold_sec, tick_size=0.25, tick_value=12.50):
    """Run sweep detection strategy with given parameters."""
    sweeps = detect_sweeps(trades, n_levels_min=n_levels_min,
                           time_window_sec=time_window_sec, tick_size=tick_size)

    if len(sweeps) < 5:
        return {
            "n_levels_min": n_levels_min,
            "time_window_sec": time_window_sec,
            "hold_sec": hold_sec,
            "n_sweeps": len(sweeps),
            "n_trades": 0,
            "sharpe": 0.0,
            "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
            "skipped": True,
        }

    pnls = compute_forward_returns(sweeps, trades, hold_sec, tick_size)
    n_trades = len(pnls)

    if n_trades < 5:
        return {
            "n_levels_min": n_levels_min,
            "time_window_sec": time_window_sec,
            "hold_sec": hold_sec,
            "n_sweeps": len(sweeps),
            "n_trades": n_trades,
            "sharpe": 0.0,
            "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
            "skipped": True,
        }

    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if pnls.std() > 0 else 1e-9
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252)
    win_rate = float((pnls > 0).mean())

    # Sweep direction breakdown
    up_sweeps = sum(1 for s in sweeps if s["direction"] == "UP")
    down_sweeps = sum(1 for s in sweeps if s["direction"] == "DOWN")

    return {
        "n_levels_min": n_levels_min,
        "time_window_sec": time_window_sec,
        "hold_sec": hold_sec,
        "n_sweeps": len(sweeps),
        "n_trades": n_trades,
        "sharpe": round(float(sharpe), 4),
        "total_pnl_ticks": round(total_pnl, 2),
        "total_pnl_usd": round(total_pnl * tick_value, 2),
        "win_rate": round(win_rate, 4),
        "avg_pnl_ticks": round(avg_pnl, 4),
        "up_sweeps": up_sweeps,
        "down_sweeps": down_sweeps,
        "avg_levels_swept": round(np.mean([s["n_levels"] for s in sweeps]), 2),
        "avg_sweep_size": round(np.mean([s["total_size"] for s in sweeps]), 1),
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Trade Sweep Detection Strategy Backtest")
    parser.add_argument("--data-dir", default="/home/jupiter/Lvl3Quant/data/",
                        help="Directory containing MBO JSONL files")
    parser.add_argument("--output", default="sweep_strategy_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--tick-size", type=float, default=0.25,
                        help="Tick size (ES=0.25)")
    parser.add_argument("--tick-value", type=float, default=12.50,
                        help="Dollar value per tick (ES=$12.50)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max files to process (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("Trade Sweep Detection Strategy")
    print("=" * 60)

    # Load MBO data
    pattern = os.path.join(args.data_dir, "*_rithmic.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try alternative patterns
        for alt in ["*.jsonl", "*mbo*.jsonl", "*trades*.jsonl"]:
            files = sorted(glob.glob(os.path.join(args.data_dir, alt)))
            if files:
                break
    if not files:
        print(f"No JSONL files found in {args.data_dir}")
        sys.exit(1)

    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} JSONL files")

    # Load all trades
    all_trades = []
    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        trades = load_mbo_trades(filepath)
        print(f"  [{i+1}/{len(files)}] {fname}: {len(trades):,} trades")
        all_trades.extend(trades)

    print(f"\nTotal trades loaded: {len(all_trades):,}")

    if len(all_trades) < 100:
        print("Insufficient trade data. Exiting.")
        sys.exit(1)

    # Sort by timestamp
    all_trades.sort(key=lambda x: x[0])

    # Parameter sweep
    n_levels_range = [3, 4, 5, 7]
    time_window_range = [0.25, 0.5, 1.0, 2.0]   # seconds
    hold_sec_range = [10, 20, 30, 60, 120]        # seconds

    total_configs = len(n_levels_range) * len(time_window_range) * len(hold_sec_range)
    print(f"\nSweeping {total_configs} configurations...")
    print(f"  n_levels_min: {n_levels_range}")
    print(f"  time_windows: {time_window_range}s")
    print(f"  hold_times: {hold_sec_range}s")

    results = []
    best_sharpe = -np.inf
    best_config = None
    t0 = time.time()
    done = 0

    for n_levels_min in n_levels_range:
        for time_window in time_window_range:
            for hold_sec in hold_sec_range:
                result = run_strategy(
                    all_trades, n_levels_min, time_window, hold_sec,
                    args.tick_size, args.tick_value
                )
                results.append(result)
                if result["sharpe"] > best_sharpe and not result.get("skipped"):
                    best_sharpe = result["sharpe"]
                    best_config = result

                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total_configs}] {elapsed:.1f}s elapsed")

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
              f"levels>={r['n_levels_min']} window={r['time_window_sec']}s hold={r['hold_sec']}s")

    # Save
    output = {
        "strategy": "trade_sweep_detection_momentum",
        "data_dir": args.data_dir,
        "n_files": len(files),
        "n_total_trades": len(all_trades),
        "tick_size": args.tick_size,
        "tick_value": args.tick_value,
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
