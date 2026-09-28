#!/usr/bin/env python3
"""
Bid-Ask Imbalance Mean Reversion Strategy
==========================================
Signal: (bid_depth - ask_depth) / (bid_depth + ask_depth) at top N levels.
When imbalance > threshold: LONG (price reverts toward heavy side).
When imbalance < -threshold: SHORT.

Uses book tensor NPZ files (shape: N_bars x 20_levels x 4_features).
Levels 0-9 = bid (best to worst), 10-19 = ask (best to worst).
Features: [0] price_relative_to_mid, [1] depth_lots (log1p), [2] num_orders (log1p), [3] queue_age_seconds (log1p).

Usage: python imbalance_strategy.py [--data-dir /path/to/npz] [--output results.json]
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np


def load_book_tensors(data_dir):
    """Load all book tensor NPZ files from directory."""
    pattern = os.path.join(data_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No NPZ files found in {data_dir}")
        sys.exit(1)

    all_data = []
    file_names = []
    for f in files:
        try:
            npz = np.load(f)
            # Try common key names
            for key in ["book", "tensor", "data", "arr_0"]:
                if key in npz:
                    arr = npz[key]
                    break
            else:
                # Use first array
                arr = npz[list(npz.keys())[0]]

            if arr.ndim == 3 and arr.shape[1] == 20 and arr.shape[2] == 4:
                all_data.append(arr)
                file_names.append(os.path.basename(f))
            else:
                print(f"  Skipping {os.path.basename(f)}: shape {arr.shape}")
        except Exception as e:
            print(f"  Error loading {os.path.basename(f)}: {e}")

    print(f"Loaded {len(all_data)} files, {sum(a.shape[0] for a in all_data)} total bars")
    return all_data, file_names


def compute_imbalance(book_tensor, n_levels):
    """
    Compute bid-ask imbalance at top N levels.

    book_tensor: (N_bars, 20, 4)
    n_levels: how many levels from each side to use (1-10)

    Returns: (N_bars,) imbalance in [-1, 1]
    """
    # depth_lots is feature index 1, stored as log1p
    # Bid levels: 0 to n_levels-1, Ask levels: 10 to 10+n_levels-1
    bid_depth_log = book_tensor[:, :n_levels, 1]       # (N, n_levels)
    ask_depth_log = book_tensor[:, 10:10+n_levels, 1]  # (N, n_levels)

    # Convert from log1p back to raw lots
    bid_depth = np.expm1(bid_depth_log)
    ask_depth = np.expm1(ask_depth_log)

    # Sum across levels
    total_bid = bid_depth.sum(axis=1)  # (N,)
    total_ask = ask_depth.sum(axis=1)  # (N,)

    denom = total_bid + total_ask
    # Avoid division by zero
    imbalance = np.where(denom > 0, (total_bid - total_ask) / denom, 0.0)
    return imbalance


def compute_mid_price(book_tensor):
    """
    Compute mid price from book tensor.
    Feature 0 is price_relative_to_mid, so mid = best_bid_price - relative + mid
    Since relative is TO mid, the mid price IS the reference. We track returns instead.

    Returns relative mid-price changes (in ticks, using bid/ask level 0 prices).
    """
    # Best bid relative price (level 0, feature 0)
    best_bid_rel = book_tensor[:, 0, 0]
    # Best ask relative price (level 10, feature 0)
    best_ask_rel = book_tensor[:, 10, 0]
    # Mid is at 0 by definition (prices are relative to mid)
    # So actual mid change = 0, but we need spread info for returns
    # The mid-to-mid return between bars is approximated by shift in the relative prices
    # Since prices are relative to current mid, mid_t - mid_{t-1} ~ -(bid_rel_t - bid_rel_{t-1}) roughly
    # Better: use the average of bid and ask relative prices
    mid_relative = (best_bid_rel + best_ask_rel) / 2.0
    return mid_relative


def run_strategy(book_data, n_levels, threshold, hold_bars, tick_value=12.50):
    """
    Run imbalance mean reversion strategy on concatenated book data.

    Parameters:
        book_data: (N, 20, 4) numpy array
        n_levels: number of book levels to use for imbalance
        threshold: imbalance threshold to trigger trade (0 to 1)
        hold_bars: how many bars to hold position
        tick_value: dollar value per tick (ES = $12.50)

    Returns: dict with strategy metrics
    """
    N = book_data.shape[0]
    if N < hold_bars + 10:
        return None

    imbalance = compute_imbalance(book_data, n_levels)

    # Use best bid/ask relative prices to compute bar-to-bar mid returns
    best_bid_rel = book_data[:, 0, 0]
    best_ask_rel = book_data[:, 10, 0]
    mid_rel = (best_bid_rel + best_ask_rel) / 2.0

    # Forward returns: mid_price change over hold_bars
    # mid_rel is price relative to current mid, so we need cumulative drift
    # Approximate: use diff of mid_rel as the per-bar return
    mid_diff = np.diff(mid_rel, prepend=mid_rel[0])
    # Cumulative sum gives us the reconstructed mid price trajectory
    mid_cumsum = np.cumsum(mid_diff)

    # Forward return = mid_cumsum[t + hold_bars] - mid_cumsum[t]
    fwd_return = np.zeros(N)
    fwd_return[:N - hold_bars] = mid_cumsum[hold_bars:] - mid_cumsum[:N - hold_bars]

    # Generate signals
    long_signal = imbalance > threshold      # heavy bid = expect price to rise
    short_signal = imbalance < -threshold    # heavy ask = expect price to fall

    # PnL per trade (in ticks, then convert to dollars)
    # Long: profit if price goes up. Short: profit if price goes down.
    long_pnl = fwd_return[long_signal]
    short_pnl = -fwd_return[short_signal]

    all_pnl = np.concatenate([long_pnl, short_pnl]) if (len(long_pnl) + len(short_pnl)) > 0 else np.array([])

    n_trades = len(all_pnl)
    if n_trades < 10:
        return {
            "n_levels": n_levels,
            "threshold": threshold,
            "hold_bars": hold_bars,
            "n_trades": n_trades,
            "sharpe": 0.0,
            "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
            "avg_pnl_ticks": 0.0,
            "long_trades": int(long_signal.sum()),
            "short_trades": int(short_signal.sum()),
            "skipped": True,
        }

    total_pnl_ticks = float(all_pnl.sum())
    avg_pnl = float(all_pnl.mean())
    std_pnl = float(all_pnl.std()) if all_pnl.std() > 0 else 1e-9
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252)  # Annualized
    win_rate = float((all_pnl > 0).mean())

    return {
        "n_levels": n_levels,
        "threshold": round(threshold, 3),
        "hold_bars": hold_bars,
        "n_trades": n_trades,
        "sharpe": round(float(sharpe), 4),
        "total_pnl_ticks": round(total_pnl_ticks, 2),
        "total_pnl_usd": round(total_pnl_ticks * tick_value, 2),
        "win_rate": round(win_rate, 4),
        "avg_pnl_ticks": round(avg_pnl, 4),
        "long_trades": int(long_signal.sum()),
        "short_trades": int(short_signal.sum()),
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Bid-Ask Imbalance Mean Reversion Backtest")
    parser.add_argument("--data-dir", default="/home/jupiter/Lvl3Quant/data/processed/dl_book_cache/",
                        help="Directory containing book tensor NPZ files")
    parser.add_argument("--output", default="imbalance_strategy_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--tick-value", type=float, default=12.50,
                        help="Dollar value per tick (ES=$12.50)")
    args = parser.parse_args()

    print("=" * 60)
    print("Bid-Ask Imbalance Mean Reversion Strategy")
    print("=" * 60)

    # Load data
    print(f"\nLoading book tensors from {args.data_dir}...")
    all_data, file_names = load_book_tensors(args.data_dir)

    # Concatenate all days
    book_data = np.concatenate(all_data, axis=0)
    print(f"Total bars: {book_data.shape[0]:,}")

    # Parameter sweep
    n_levels_range = [1, 3, 5, 10]
    threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # hold_bars: depends on bar frequency. Assuming ~100ms bars (10 bars/sec):
    # 10s = 100 bars, 30s = 300 bars, 60s = 600 bars
    # If bars are 1s: 10s = 10 bars, etc.
    # We'll sweep a range and let the user interpret based on their bar freq
    hold_bars_range = [10, 30, 50, 100, 300, 600]

    total_configs = len(n_levels_range) * len(threshold_range) * len(hold_bars_range)
    print(f"\nSweeping {total_configs} configurations...")
    print(f"  n_levels: {n_levels_range}")
    print(f"  thresholds: {threshold_range}")
    print(f"  hold_bars: {hold_bars_range}")

    results = []
    best_sharpe = -np.inf
    best_config = None
    t0 = time.time()

    for i, n_levels in enumerate(n_levels_range):
        # Pre-compute imbalance for this n_levels (avoid redundant computation)
        imbalance = compute_imbalance(book_data, n_levels)

        for threshold in threshold_range:
            for hold_bars in hold_bars_range:
                result = run_strategy(book_data, n_levels, threshold, hold_bars, args.tick_value)
                if result is not None:
                    results.append(result)
                    if result["sharpe"] > best_sharpe and not result.get("skipped"):
                        best_sharpe = result["sharpe"]
                        best_config = result

        elapsed = time.time() - t0
        done = (i + 1) * len(threshold_range) * len(hold_bars_range)
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{total_configs}] n_levels={n_levels} done ({rate:.0f} configs/sec)")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} configs in {elapsed:.1f}s")

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: Sharpe={r['sharpe']:+.3f}  PnL=${r['total_pnl_usd']:+,.0f}  "
              f"WR={r['win_rate']:.1%}  Trades={r['n_trades']:,}  "
              f"levels={r['n_levels']} thresh={r['threshold']} hold={r['hold_bars']}")

    # Save results
    output = {
        "strategy": "bid_ask_imbalance_mean_reversion",
        "data_dir": args.data_dir,
        "n_files": len(file_names),
        "n_bars": int(book_data.shape[0]),
        "tick_value": args.tick_value,
        "n_configs": len(results),
        "best_config": best_config,
        "top_20": results[:20],
        "all_results": results,
        "files_used": file_names,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
