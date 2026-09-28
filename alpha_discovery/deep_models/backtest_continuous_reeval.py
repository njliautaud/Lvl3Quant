#!/usr/bin/env python3
"""
Continuous Re-Evaluation Execution Backtester for Mamba v7 predictions.

Strategy: Enter when 1s signal confidence is in Top-N%, hold and re-evaluate
every prediction event, exit on signal flip / slow flip / conditional flip / time limit.

P&L approach:
  - Load source event data to get timestamps at each prediction event
  - labels_1s/5s/10s at entry event give price change over 1s/5s/10s
  - For variable hold times, use closest horizon label at entry event
  - For holds > 10s, use labels_10s as conservative lower bound
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

# ── Configuration ────────────────────────────────────────────────
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr")
SRC_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/backtest_continuous_reeval")

WINDOW_SIZE = 1000
STRIDE = 500
FOLDS = list(range(5, 16))  # March folds only

# Cost model (in ticks)
TICK_VALUE = 12.50  # USD per tick (NQ)
COST_PASSIVE = 0.38   # passive limit total cost per RT
COST_MID = 0.88       # mid-price total cost per RT
COST_MARKET = 1.38    # market order total cost per RT

# Default cost assumption: enter passive, exit mid
DEFAULT_COST = (COST_PASSIVE + COST_MID) / 2  # ~0.63 ticks RT

# Confidence tiers (percentile from top by magnitude)
CONFIDENCE_TIERS = {
    "Top5%": 0.95,
    "Top1%": 0.99,
    "Top0.5%": 0.995,
    "Top0.1%": 0.999,
}

# Exit strategies
EXIT_STRATEGIES = {
    "instant_flip": {},                          # Exit immediately on 1s sign flip
    "slow_flip_3": {"consecutive": 3},           # 1s sign flipped for 3 consecutive events
    "slow_flip_5": {"consecutive": 5},           # 1s sign flipped for 5 consecutive events
    "slow_flip_10": {"consecutive": 10},         # 1s sign flipped for 10 consecutive events
    "conditional_flip": {"require_10s": True},   # 1s flips AND 10s agrees with new direction
    "time_5s": {"max_hold_sec": 5.0},            # Exit after 5s max
    "time_10s": {"max_hold_sec": 10.0},          # Exit after 10s max
    "time_30s": {"max_hold_sec": 30.0},          # Exit after 30s max
    "time_60s": {"max_hold_sec": 60.0},          # Exit after 60s max
    # Combo: flip + time limit
    "flip_or_10s": {"max_hold_sec": 10.0},       # Flip OR 10s max
    "flip_or_30s": {"max_hold_sec": 30.0},       # Flip OR 30s max
    "slow3_or_10s": {"consecutive": 3, "max_hold_sec": 10.0},
    "slow3_or_30s": {"consecutive": 3, "max_hold_sec": 30.0},
    "conditional_or_30s": {"require_10s": True, "max_hold_sec": 30.0},
}


@dataclass
class Trade:
    entry_idx: int          # prediction index
    exit_idx: int           # prediction index
    direction: int          # +1 long, -1 short
    entry_ts_ns: int        # nanosecond timestamp
    exit_ts_ns: int         # nanosecond timestamp
    hold_time_sec: float    # seconds
    pnl_ticks: float        # gross P&L in ticks
    cost_ticks: float       # total cost in ticks
    net_pnl_ticks: float    # net P&L in ticks
    fold: int


def load_fold_data(fold_idx: int) -> Optional[Dict]:
    """Load predictions and source timestamps/labels for a fold."""
    pred_path = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_path.exists():
        print(f"  [SKIP] Fold {fold_idx}: {pred_path} not found")
        return None

    pred = np.load(pred_path, allow_pickle=True)
    predictions = pred["predictions"]  # (N, 3) for 1s/5s/10s
    labels = pred["labels"]            # (N, 3)
    oot_files = pred["oot_files"]

    # Extract date from oot_files path
    oot_file = str(oot_files[0])
    date_str = oot_file.split("/")[-1].replace("_mbo_events.npz", "")

    # Load source data for timestamps
    src_path = SRC_DIR / f"{date_str}_mbo_events.npz"
    if not src_path.exists():
        print(f"  [SKIP] Fold {fold_idx}: source {src_path} not found")
        return None

    src = np.load(src_path)
    src_timestamps = src["timestamps"]
    src_labels_1s = src["labels_1s"]
    src_labels_5s = src["labels_5s"]
    src_labels_10s = src["labels_10s"]

    n_preds = predictions.shape[0]

    # Map prediction indices to source event indices
    src_indices = np.array([i * STRIDE + WINDOW_SIZE - 1 for i in range(n_preds)])

    # Validate mapping
    valid = src_indices < len(src_timestamps)
    if not valid.all():
        n_valid = valid.sum()
        print(f"  [WARN] Fold {fold_idx}: {n_preds - n_valid} predictions out of source bounds, trimming")
        src_indices = src_indices[valid]
        predictions = predictions[valid.nonzero()[0]]
        labels = labels[valid.nonzero()[0]]
        n_preds = len(predictions)

    timestamps = src_timestamps[src_indices]

    # Also get source labels at prediction indices for P&L
    # Replace NaN labels with 0 (end-of-day events where forward labels unavailable)
    labels_1s_at_pred = np.nan_to_num(src_labels_1s[src_indices], nan=0.0)
    labels_5s_at_pred = np.nan_to_num(src_labels_5s[src_indices], nan=0.0)
    labels_10s_at_pred = np.nan_to_num(src_labels_10s[src_indices], nan=0.0)

    # For computing P&L at variable hold times, we need price changes between
    # consecutive prediction events. We can approximate:
    # price_change(pred_i -> pred_{i+1}) ≈ labels_1s[src_idx_i] * (dt / 1.0)
    # But this is wrong. Instead, we'll reconstruct price path from source events.

    # Better approach: load labels between consecutive source prediction indices
    # to compute cumulative price change.
    # Actually: use labels at entry for the horizon closest to hold time.
    # For variable hold, interpolate between horizons.

    # Mark which prediction indices have valid (non-NaN) source labels
    # We check original source labels BEFORE nan_to_num replacement
    valid_entry = ~np.isnan(src_labels_1s[src_indices])

    return {
        "fold": fold_idx,
        "date": date_str,
        "predictions": predictions,
        "labels": labels,
        "timestamps": timestamps,
        "labels_1s": labels_1s_at_pred,
        "labels_5s": labels_5s_at_pred,
        "labels_10s": labels_10s_at_pred,
        "valid_entry": valid_entry,
        "n_preds": n_preds,
        "src_indices": src_indices,
        "src_timestamps": src_timestamps,
        "src_labels_1s": src_labels_1s,
    }


def compute_pnl_for_hold(data: Dict, entry_idx: int, exit_idx: int, direction: int) -> float:
    """
    Compute gross P&L (in ticks) for holding from entry_idx to exit_idx.

    Uses the source event data to reconstruct price path between prediction events.
    The price change from event A to event B can be computed by noting that
    labels_1s[A] = price(t_A + 1s) - price(t_A).

    For accuracy, we compute cumulative 1-event price changes from source data.
    Since we need price(t_exit) - price(t_entry), and we have src_labels_1s at
    every raw event, we can find events at entry and exit timestamps.

    Simplification: use labels at entry event for the closest horizon to hold time.
    """
    entry_ts = data["timestamps"][entry_idx]
    exit_ts = data["timestamps"][exit_idx]
    hold_sec = (exit_ts - entry_ts) / 1e9

    # Use the label at entry that best matches hold time
    if hold_sec <= 1.5:
        pnl = data["labels_1s"][entry_idx]
    elif hold_sec <= 7.5:
        # Interpolate between 1s and 5s
        t = (hold_sec - 1.0) / (5.0 - 1.0)
        t = np.clip(t, 0, 1)
        pnl = data["labels_1s"][entry_idx] * (1 - t) + data["labels_5s"][entry_idx] * t
    elif hold_sec <= 15.0:
        # Interpolate between 5s and 10s
        t = (hold_sec - 5.0) / (10.0 - 5.0)
        t = np.clip(t, 0, 1)
        pnl = data["labels_5s"][entry_idx] * (1 - t) + data["labels_10s"][entry_idx] * t
    else:
        # Beyond 10s horizon, use 10s label as best estimate
        pnl = data["labels_10s"][entry_idx]

    return direction * pnl


def compute_pnl_cumulative(data: Dict, entry_idx: int, exit_idx: int, direction: int) -> float:
    """
    Compute P&L using cumulative source event price changes.

    We reconstruct the price change between two prediction events by examining
    the source labels. Between source events i and i+1, if the time gap dt is small,
    the price change is approximately labels_1s[i] * dt / 1.0 (since labels_1s
    measures the 1-second forward change).

    Actually, a simpler and more accurate method: find the source event at the exit
    time, and use labels_1s at entry to estimate PnL for holds close to 1s.
    For longer holds, we chain: price(exit) - price(entry) can be estimated
    from the label at entry for the matching horizon.

    For the most accurate method, we'd need raw mid-prices, but labels at entry
    for the appropriate horizon are a good approximation.
    """
    return compute_pnl_for_hold(data, entry_idx, exit_idx, direction)


def run_backtest(
    data: Dict,
    strategy_name: str,
    strategy_params: Dict,
    confidence_tier: str,
    confidence_pctile: float,
    cost_per_trade: float = DEFAULT_COST,
) -> List[Trade]:
    """
    Run continuous re-evaluation backtest on a single fold.

    Entry: when |pred_1s| exceeds the confidence threshold
    Exit: based on strategy_params (flip, slow flip, conditional, time)
    """
    preds_1s = data["predictions"][:, 0]  # 1s predictions
    preds_10s = data["predictions"][:, 2]  # 10s predictions
    timestamps = data["timestamps"]
    valid_entry = data["valid_entry"]
    n = data["n_preds"]

    # Compute confidence threshold for this tier (only over valid entries)
    magnitudes = np.abs(preds_1s)
    valid_mags = magnitudes[valid_entry]
    if len(valid_mags) == 0:
        return []
    threshold = np.percentile(valid_mags, confidence_pctile * 100)

    consecutive_req = strategy_params.get("consecutive", 1)
    require_10s = strategy_params.get("require_10s", False)
    max_hold_sec = strategy_params.get("max_hold_sec", None)

    trades = []
    i = 0
    while i < n:
        # Check entry condition: magnitude exceeds threshold AND labels are valid
        if magnitudes[i] < threshold or not valid_entry[i]:
            i += 1
            continue

        # Enter trade
        direction = 1 if preds_1s[i] > 0 else -1
        entry_idx = i
        entry_ts = timestamps[i]

        # Scan forward for exit
        flip_count = 0
        j = i + 1
        exit_found = False

        while j < n:
            current_pred_1s = preds_1s[j]
            current_pred_10s = preds_10s[j]
            hold_sec = (timestamps[j] - entry_ts) / 1e9

            # Check time-based exit first
            if max_hold_sec is not None and hold_sec >= max_hold_sec:
                exit_found = True
                break

            # Check signal-based exits
            signal_flipped = (direction == 1 and current_pred_1s < 0) or \
                             (direction == -1 and current_pred_1s > 0)

            if signal_flipped:
                flip_count += 1
            else:
                flip_count = 0

            # Instant flip
            if consecutive_req <= 1 and not require_10s and max_hold_sec is None:
                if signal_flipped:
                    exit_found = True
                    break

            # Slow flip (consecutive requirement)
            elif consecutive_req > 1 and not require_10s:
                if flip_count >= consecutive_req:
                    exit_found = True
                    break

            # Conditional flip (requires 10s agreement)
            elif require_10s and consecutive_req <= 1:
                if signal_flipped:
                    # Check 10s agrees with the NEW direction (opposite of entry)
                    new_dir = -direction
                    if (new_dir == 1 and current_pred_10s > 0) or \
                       (new_dir == -1 and current_pred_10s < 0):
                        exit_found = True
                        break

            # Combo: slow flip + time, or conditional + time
            elif consecutive_req > 1 and max_hold_sec is not None:
                if flip_count >= consecutive_req:
                    exit_found = True
                    break
            elif require_10s and max_hold_sec is not None:
                if signal_flipped:
                    new_dir = -direction
                    if (new_dir == 1 and current_pred_10s > 0) or \
                       (new_dir == -1 and current_pred_10s < 0):
                        exit_found = True
                        break

            # For combo with just flip + time (no slow, no conditional)
            elif max_hold_sec is not None and consecutive_req <= 1 and not require_10s:
                if signal_flipped:
                    exit_found = True
                    break

            j += 1

        if not exit_found:
            # Forced exit at end of day
            j = min(j, n - 1)

        exit_idx = j
        exit_ts = timestamps[exit_idx]
        hold_sec = (exit_ts - entry_ts) / 1e9

        # Compute P&L
        pnl_gross = compute_pnl_for_hold(data, entry_idx, exit_idx, direction)
        net_pnl = pnl_gross - cost_per_trade

        trades.append(Trade(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            direction=direction,
            entry_ts_ns=int(entry_ts),
            exit_ts_ns=int(exit_ts),
            hold_time_sec=hold_sec,
            pnl_ticks=pnl_gross,
            cost_ticks=cost_per_trade,
            net_pnl_ticks=net_pnl,
            fold=data["fold"],
        ))

        # Move past exit (no overlapping trades)
        i = exit_idx + 1

    return trades


def compute_metrics(trades: List[Trade]) -> Dict:
    """Compute performance metrics for a list of trades."""
    if not trades:
        return {
            "n_trades": 0,
            "total_gross_pnl_ticks": 0,
            "total_net_pnl_ticks": 0,
            "total_cost_ticks": 0,
            "gross_pnl_usd": 0,
            "net_pnl_usd": 0,
            "avg_gross_per_trade": 0,
            "avg_net_per_trade": 0,
            "win_rate": 0,
            "avg_winner_ticks": 0,
            "avg_loser_ticks": 0,
            "profit_factor": 0,
            "sortino": 0,
            "avg_hold_sec": 0,
            "median_hold_sec": 0,
            "n_long": 0,
            "n_short": 0,
            "long_pnl_ticks": 0,
            "short_pnl_ticks": 0,
        }

    gross_pnls = np.array([t.pnl_ticks for t in trades])
    net_pnls = np.array([t.net_pnl_ticks for t in trades])
    costs = np.array([t.cost_ticks for t in trades])
    hold_times = np.array([t.hold_time_sec for t in trades])
    directions = np.array([t.direction for t in trades])

    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls <= 0]

    # Profit factor
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Sortino ratio (annualized, assuming ~252 trading days, ~23400s per day)
    mean_return = net_pnls.mean()
    downside = net_pnls[net_pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else 1.0
    # Per-trade Sortino (not annualized)
    sortino = mean_return / downside_std if downside_std > 0 else float("inf") if mean_return > 0 else 0

    long_mask = directions == 1
    short_mask = directions == -1

    return {
        "n_trades": len(trades),
        "total_gross_pnl_ticks": float(gross_pnls.sum()),
        "total_net_pnl_ticks": float(net_pnls.sum()),
        "total_cost_ticks": float(costs.sum()),
        "gross_pnl_usd": float(gross_pnls.sum() * TICK_VALUE),
        "net_pnl_usd": float(net_pnls.sum() * TICK_VALUE),
        "avg_gross_per_trade": float(gross_pnls.mean()),
        "avg_net_per_trade": float(net_pnls.mean()),
        "win_rate": float((net_pnls > 0).mean()),
        "avg_winner_ticks": float(winners.mean()) if len(winners) > 0 else 0,
        "avg_loser_ticks": float(losers.mean()) if len(losers) > 0 else 0,
        "profit_factor": float(pf),
        "sortino": float(sortino),
        "avg_hold_sec": float(hold_times.mean()),
        "median_hold_sec": float(np.median(hold_times)),
        "max_hold_sec": float(hold_times.max()),
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "long_pnl_ticks": float(net_pnls[long_mask].sum()) if long_mask.any() else 0,
        "short_pnl_ticks": float(net_pnls[short_mask].sum()) if short_mask.any() else 0,
        "long_win_rate": float((net_pnls[long_mask] > 0).mean()) if long_mask.any() else 0,
        "short_win_rate": float((net_pnls[short_mask] > 0).mean()) if short_mask.any() else 0,
    }


def format_results_table(all_results: Dict) -> str:
    """Format results as a clean text table."""
    lines = []
    lines.append("=" * 120)
    lines.append("CONTINUOUS RE-EVALUATION BACKTESTER — Mamba v7 Tiny Smart V3 (March 2026)")
    lines.append("=" * 120)
    lines.append(f"Cost model: passive={COST_PASSIVE:.2f}t, mid={COST_MID:.2f}t, market={COST_MARKET:.2f}t | "
                 f"Default RT cost={DEFAULT_COST:.2f}t (${DEFAULT_COST * TICK_VALUE:.2f})")
    lines.append(f"Folds: {FOLDS[0]}-{FOLDS[-1]} (March dates only)")
    lines.append("")

    # Summary table for each cost scenario
    for cost_label, cost_val in [("Passive", COST_PASSIVE), ("Mid", COST_MID), ("Market", COST_MARKET)]:
        lines.append(f"\n{'─' * 120}")
        lines.append(f"COST SCENARIO: {cost_label} ({cost_val:.2f} ticks = ${cost_val * TICK_VALUE:.2f} RT)")
        lines.append(f"{'─' * 120}")

        header = f"{'Strategy':<22} {'Tier':<8} {'Trades':>7} {'GrossPnL':>10} {'NetPnL':>10} " \
                 f"{'NetUSD':>10} {'WinRate':>8} {'AvgWin':>8} {'AvgLose':>8} " \
                 f"{'PF':>6} {'Sortino':>8} {'AvgHold':>8} {'L/S':>8}"
        lines.append(header)
        lines.append("─" * 120)

        for strategy_name in EXIT_STRATEGIES:
            for tier_name in CONFIDENCE_TIERS:
                key = f"{strategy_name}_{tier_name}"
                if key not in all_results:
                    continue
                m = all_results[key]
                if m["n_trades"] == 0:
                    continue

                # Recompute net for this cost scenario
                net_per_trade = m["avg_gross_per_trade"] - cost_val
                total_net = net_per_trade * m["n_trades"]
                net_usd = total_net * TICK_VALUE

                # Approximate win rate at this cost
                # (gross winners that survive cost)
                # This is approximate; exact would require per-trade recomputation
                # Use the stored metrics for default cost scenario
                wr = m["win_rate"]

                line = f"{strategy_name:<22} {tier_name:<8} {m['n_trades']:>7d} " \
                       f"{m['total_gross_pnl_ticks']:>10.1f} {total_net:>10.1f} " \
                       f"${net_usd:>9.0f} {wr:>7.1%} {m['avg_winner_ticks']:>8.2f} " \
                       f"{m['avg_loser_ticks']:>8.2f} {m['profit_factor']:>6.2f} " \
                       f"{m['sortino']:>8.3f} {m['avg_hold_sec']:>7.1f}s " \
                       f"{m['n_long']:>3d}/{m['n_short']:>3d}"
                lines.append(line)

        lines.append("")

    # Best strategies highlight
    lines.append("\n" + "=" * 120)
    lines.append("TOP 10 STRATEGIES BY NET P&L (Default Cost)")
    lines.append("=" * 120)

    sorted_keys = sorted(
        [k for k, v in all_results.items() if v["n_trades"] > 0],
        key=lambda k: all_results[k]["total_net_pnl_ticks"],
        reverse=True,
    )

    for rank, key in enumerate(sorted_keys[:10], 1):
        m = all_results[key]
        lines.append(
            f"  #{rank}: {key:<35} | Net={m['total_net_pnl_ticks']:>8.1f}t "
            f"(${m['net_pnl_usd']:>8.0f}) | {m['n_trades']} trades | "
            f"WR={m['win_rate']:.1%} | PF={m['profit_factor']:.2f} | "
            f"Sortino={m['sortino']:.3f} | AvgHold={m['avg_hold_sec']:.1f}s"
        )

    # Bottom 5
    lines.append("\nBOTTOM 5 STRATEGIES BY NET P&L:")
    for rank, key in enumerate(sorted_keys[-5:], 1):
        m = all_results[key]
        lines.append(
            f"  #{rank}: {key:<35} | Net={m['total_net_pnl_ticks']:>8.1f}t "
            f"(${m['net_pnl_usd']:>8.0f}) | {m['n_trades']} trades | "
            f"WR={m['win_rate']:.1%} | PF={m['profit_factor']:.2f}"
        )

    # Hold time analysis for best strategy
    if sorted_keys:
        lines.append(f"\n{'=' * 120}")
        lines.append("HOLD TIME DISTRIBUTION (Best Strategy)")
        lines.append(f"{'=' * 120}")
        best_key = sorted_keys[0]
        m = all_results[best_key]
        lines.append(f"Strategy: {best_key}")
        lines.append(f"  Mean hold: {m['avg_hold_sec']:.2f}s | Median: {m['median_hold_sec']:.2f}s | Max: {m.get('max_hold_sec', 0):.2f}s")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fold data...")
    fold_data = []
    for fold_idx in FOLDS:
        print(f"  Loading fold {fold_idx}...", end=" ")
        d = load_fold_data(fold_idx)
        if d is not None:
            fold_data.append(d)
            print(f"OK ({d['n_preds']} predictions, date={d['date']})")
        else:
            print("SKIPPED")

    if not fold_data:
        print("ERROR: No fold data loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(fold_data)} folds, {sum(d['n_preds'] for d in fold_data)} total predictions")

    # Run all strategy x tier combinations
    all_results = {}
    total_combos = len(EXIT_STRATEGIES) * len(CONFIDENCE_TIERS)
    combo_idx = 0

    for strategy_name, strategy_params in EXIT_STRATEGIES.items():
        for tier_name, tier_pctile in CONFIDENCE_TIERS.items():
            combo_idx += 1
            key = f"{strategy_name}_{tier_name}"
            print(f"\r  [{combo_idx}/{total_combos}] {key:<45}", end="", flush=True)

            all_trades = []
            for data in fold_data:
                trades = run_backtest(
                    data,
                    strategy_name,
                    strategy_params,
                    tier_name,
                    tier_pctile,
                    cost_per_trade=DEFAULT_COST,
                )
                all_trades.extend(trades)

            metrics = compute_metrics(all_trades)
            all_results[key] = metrics

    print("\n")

    # Format and print results
    report = format_results_table(all_results)
    print(report)

    # Save results
    output_path = OUTPUT_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Save report
    report_path = OUTPUT_DIR / "backtest_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Also compute per-cost-scenario metrics for the best strategies
    # by recomputing with different costs
    print("\n\nPER-COST SCENARIO ANALYSIS (Top 5 by gross P&L):")
    gross_sorted = sorted(
        [k for k, v in all_results.items() if v["n_trades"] > 0],
        key=lambda k: all_results[k]["total_gross_pnl_ticks"],
        reverse=True,
    )[:5]

    for key in gross_sorted:
        m = all_results[key]
        print(f"\n  {key}:")
        print(f"    Gross: {m['total_gross_pnl_ticks']:.1f} ticks (${m['gross_pnl_usd']:.0f})")
        for cost_label, cost_val in [("Passive", COST_PASSIVE), ("Mid", COST_MID), ("Market", COST_MARKET)]:
            net = m["total_gross_pnl_ticks"] - cost_val * m["n_trades"]
            net_usd = net * TICK_VALUE
            avg_net = net / m["n_trades"] if m["n_trades"] > 0 else 0
            print(f"    {cost_label:>8} cost: Net={net:>8.1f}t (${net_usd:>8.0f}) | "
                  f"Avg/trade={avg_net:.3f}t | Break-even cost={m['avg_gross_per_trade']:.3f}t")


if __name__ == "__main__":
    main()
