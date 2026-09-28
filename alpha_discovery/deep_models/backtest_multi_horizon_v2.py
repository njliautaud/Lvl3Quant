#!/usr/bin/env python3
"""
Multi-Horizon Continuous Re-Evaluation Backtester V2 — Mamba v7

Key improvements over v1 (backtest_continuous_reeval.py):
  - Exit strategies use 5s/10s predictions for flip detection (not just 1s)
  - New strategies: minimum hold times, strength decay, trailing strength,
    slow 10s flips, adaptive horizon
  - Reports avg_hold_events, avg_trade_count_per_day alongside other metrics
  - Full cost-scenario sweep (passive/mid/market) in results JSON

P&L approach (APPROXIMATION — same as v1):
  Uses labels at entry event for the horizon closest to actual hold time.
  For holds between horizons, interpolates. For holds > 10s, uses labels_10s.
  This is approximate but reasonable for holds under ~60s.
  Exact P&L would require raw mid-price series, which we don't store.
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
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/backtest_multi_horizon_v2")

WINDOW_SIZE = 1000
STRIDE = 500
FOLDS = list(range(5, 16))  # March folds only

# Cost model (in ticks, round-trip)
TICK_VALUE = 12.50  # USD per tick (NQ)
COST_PASSIVE = 0.38
COST_MID = 0.88
COST_MARKET = 1.38

COST_SCENARIOS = {
    "passive": COST_PASSIVE,
    "mid": COST_MID,
    "market": COST_MARKET,
}

# Confidence tiers (percentile threshold)
CONFIDENCE_TIERS = {
    "Top5%": 0.95,
    "Top1%": 0.99,
    "Top0.5%": 0.995,
    "Top0.1%": 0.999,
}

# Exit strategies — each has a name and params dict
EXIT_STRATEGIES = {
    # ── V1 strategies (kept for comparison) ──
    "instant_flip":       {"signal": "1s", "consecutive": 1},
    "slow_flip_3":        {"signal": "1s", "consecutive": 3},
    "slow_flip_5":        {"signal": "1s", "consecutive": 5},
    # ── V2 new strategies ──
    "10s_flip":           {"signal": "10s", "consecutive": 1},
    "5s_flip":            {"signal": "5s", "consecutive": 1},
    "multi_horizon_agree": {"signal": "multi", "consecutive": 1},  # exit when 1s AND 10s both flip
    "10s_flip_min5s":     {"signal": "10s", "consecutive": 1, "min_hold_sec": 5.0},
    "10s_flip_min10s":    {"signal": "10s", "consecutive": 1, "min_hold_sec": 10.0},
    "strength_decay":     {"signal": "strength_decay"},
    "trailing_strength":  {"signal": "trailing_strength"},
    "10s_flip_slow3":     {"signal": "10s", "consecutive": 3},
    "10s_flip_slow5":     {"signal": "10s", "consecutive": 5},
    "entry_1s_exit_10s":  {"signal": "10s", "consecutive": 1},  # same logic as 10s_flip (entry is always 1s)
    "adaptive_horizon":   {"signal": "adaptive"},
}


@dataclass
class Trade:
    entry_idx: int          # prediction index
    exit_idx: int           # prediction index
    direction: int          # +1 long, -1 short
    entry_ts_ns: int        # nanosecond timestamp
    exit_ts_ns: int         # nanosecond timestamp
    hold_time_sec: float
    hold_events: int        # exit_idx - entry_idx
    pnl_ticks: float        # gross P&L in ticks
    fold: int
    date: str


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

    oot_file = str(oot_files[0])
    date_str = oot_file.split("/")[-1].replace("_mbo_events.npz", "")

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
    src_indices = np.array([i * STRIDE + WINDOW_SIZE - 1 for i in range(n_preds)])

    valid = src_indices < len(src_timestamps)
    if not valid.all():
        n_valid = valid.sum()
        print(f"  [WARN] Fold {fold_idx}: {n_preds - n_valid} predictions out of bounds, trimming")
        mask = valid.nonzero()[0]
        src_indices = src_indices[mask]
        predictions = predictions[mask]
        labels = labels[mask]
        n_preds = len(predictions)

    timestamps = src_timestamps[src_indices]

    labels_1s_at_pred = np.nan_to_num(src_labels_1s[src_indices], nan=0.0)
    labels_5s_at_pred = np.nan_to_num(src_labels_5s[src_indices], nan=0.0)
    labels_10s_at_pred = np.nan_to_num(src_labels_10s[src_indices], nan=0.0)

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
    }


def compute_pnl_for_hold(data: Dict, entry_idx: int, exit_idx: int, direction: int) -> float:
    """
    Compute gross P&L (in ticks) for holding from entry_idx to exit_idx.

    APPROXIMATION: Uses labels at entry event for the closest horizon to hold time.
    Interpolates between horizons for intermediate hold times.
    For holds > 10s, uses labels_10s (conservative).
    Exact P&L would require raw mid-price series.
    """
    entry_ts = data["timestamps"][entry_idx]
    exit_ts = data["timestamps"][exit_idx]
    hold_sec = (exit_ts - entry_ts) / 1e9

    if hold_sec <= 0:
        return 0.0

    if hold_sec <= 1.5:
        pnl = data["labels_1s"][entry_idx]
    elif hold_sec <= 7.5:
        t = (hold_sec - 1.0) / (5.0 - 1.0)
        t = np.clip(t, 0, 1)
        pnl = data["labels_1s"][entry_idx] * (1 - t) + data["labels_5s"][entry_idx] * t
    elif hold_sec <= 15.0:
        t = (hold_sec - 5.0) / (10.0 - 5.0)
        t = np.clip(t, 0, 1)
        pnl = data["labels_5s"][entry_idx] * (1 - t) + data["labels_10s"][entry_idx] * t
    else:
        # Beyond 10s horizon — use 10s label as best available estimate
        pnl = data["labels_10s"][entry_idx]

    return direction * pnl


def check_exit(
    data: Dict,
    entry_idx: int,
    j: int,
    direction: int,
    strategy_params: Dict,
    flip_count: int,
    entry_strength: float,
    strength_p50: float,
) -> Tuple[bool, int]:
    """
    Check if exit condition is met at prediction event j.

    Returns (should_exit, updated_flip_count).
    """
    preds_1s = data["predictions"][:, 0]
    preds_5s = data["predictions"][:, 1]
    preds_10s = data["predictions"][:, 2]
    timestamps = data["timestamps"]

    signal_type = strategy_params.get("signal", "1s")
    consecutive_req = strategy_params.get("consecutive", 1)
    min_hold_sec = strategy_params.get("min_hold_sec", 0.0)

    hold_sec = (timestamps[j] - timestamps[entry_idx]) / 1e9

    # Enforce minimum hold time
    if hold_sec < min_hold_sec:
        return False, 0

    # ── Signal-specific exit logic ──

    if signal_type == "1s":
        flipped = (direction == 1 and preds_1s[j] < 0) or (direction == -1 and preds_1s[j] > 0)
        if flipped:
            flip_count += 1
        else:
            flip_count = 0
        if flip_count >= consecutive_req:
            return True, flip_count
        return False, flip_count

    elif signal_type == "5s":
        flipped = (direction == 1 and preds_5s[j] < 0) or (direction == -1 and preds_5s[j] > 0)
        if flipped:
            flip_count += 1
        else:
            flip_count = 0
        if flip_count >= consecutive_req:
            return True, flip_count
        return False, flip_count

    elif signal_type == "10s":
        flipped = (direction == 1 and preds_10s[j] < 0) or (direction == -1 and preds_10s[j] > 0)
        if flipped:
            flip_count += 1
        else:
            flip_count = 0
        if flip_count >= consecutive_req:
            return True, flip_count
        return False, flip_count

    elif signal_type == "multi":
        # Exit when BOTH 1s AND 10s agree on opposite direction
        flip_1s = (direction == 1 and preds_1s[j] < 0) or (direction == -1 and preds_1s[j] > 0)
        flip_10s = (direction == 1 and preds_10s[j] < 0) or (direction == -1 and preds_10s[j] > 0)
        if flip_1s and flip_10s:
            return True, 0
        return False, 0

    elif signal_type == "strength_decay":
        # Exit when 1s prediction magnitude drops below 50th percentile
        if abs(preds_1s[j]) < strength_p50:
            return True, 0
        return False, 0

    elif signal_type == "trailing_strength":
        # Exit when 1s prediction magnitude drops below 50% of entry strength
        if abs(preds_1s[j]) < 0.5 * entry_strength:
            return True, 0
        return False, 0

    elif signal_type == "adaptive":
        # Hold while 5s and 10s agree with entry direction.
        # Exit when 10s flips, even if 1s still agrees.
        flip_10s = (direction == 1 and preds_10s[j] < 0) or (direction == -1 and preds_10s[j] > 0)
        if flip_10s:
            return True, 0
        return False, 0

    return False, flip_count


def run_backtest(
    data: Dict,
    strategy_name: str,
    strategy_params: Dict,
    confidence_pctile: float,
) -> List[Trade]:
    """
    Run continuous re-evaluation backtest on a single fold.

    Entry: when |pred_1s| exceeds the confidence threshold.
    Exit: based on strategy-specific logic.
    """
    preds_1s = data["predictions"][:, 0]
    timestamps = data["timestamps"]
    valid_entry = data["valid_entry"]
    n = data["n_preds"]

    magnitudes = np.abs(preds_1s)
    valid_mags = magnitudes[valid_entry]
    if len(valid_mags) == 0:
        return []

    threshold = np.percentile(valid_mags, confidence_pctile * 100)

    # Pre-compute 50th percentile of 1s magnitude for strength_decay strategy
    strength_p50 = np.percentile(valid_mags, 50)

    trades = []
    i = 0
    while i < n:
        if magnitudes[i] < threshold or not valid_entry[i]:
            i += 1
            continue

        direction = 1 if preds_1s[i] > 0 else -1
        entry_idx = i
        entry_ts = timestamps[i]
        entry_strength = magnitudes[i]

        flip_count = 0
        j = i + 1
        exit_found = False

        while j < n:
            should_exit, flip_count = check_exit(
                data, entry_idx, j, direction, strategy_params,
                flip_count, entry_strength, strength_p50,
            )
            if should_exit:
                exit_found = True
                break
            j += 1

        if not exit_found:
            j = min(j, n - 1)

        exit_idx = j
        exit_ts = timestamps[exit_idx]
        hold_sec = (exit_ts - entry_ts) / 1e9
        hold_events = exit_idx - entry_idx

        pnl_gross = compute_pnl_for_hold(data, entry_idx, exit_idx, direction)

        trades.append(Trade(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            direction=direction,
            entry_ts_ns=int(entry_ts),
            exit_ts_ns=int(exit_ts),
            hold_time_sec=hold_sec,
            hold_events=hold_events,
            pnl_ticks=pnl_gross,
            fold=data["fold"],
            date=data["date"],
        ))

        i = exit_idx + 1

    return trades


def compute_metrics(trades: List[Trade], cost_per_trade: float) -> Dict:
    """Compute performance metrics for a list of trades at a given cost."""
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
            "max_hold_sec": 0,
            "avg_hold_events": 0,
            "n_long": 0,
            "n_short": 0,
            "long_pnl_ticks": 0,
            "short_pnl_ticks": 0,
            "long_win_rate": 0,
            "short_win_rate": 0,
            "avg_trade_count_per_day": 0,
        }

    gross_pnls = np.array([t.pnl_ticks for t in trades])
    net_pnls = gross_pnls - cost_per_trade
    hold_times = np.array([t.hold_time_sec for t in trades])
    hold_events = np.array([t.hold_events for t in trades])
    directions = np.array([t.direction for t in trades])

    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls <= 0]

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    mean_return = net_pnls.mean()
    downside = net_pnls[net_pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else 1.0
    sortino = mean_return / downside_std if downside_std > 0 else (float("inf") if mean_return > 0 else 0)

    long_mask = directions == 1
    short_mask = directions == -1

    # Trades per day
    unique_dates = set(t.date for t in trades)
    n_days = max(len(unique_dates), 1)

    return {
        "n_trades": len(trades),
        "total_gross_pnl_ticks": float(gross_pnls.sum()),
        "total_net_pnl_ticks": float(net_pnls.sum()),
        "total_cost_ticks": float(cost_per_trade * len(trades)),
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
        "avg_hold_events": float(hold_events.mean()),
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "long_pnl_ticks": float(net_pnls[long_mask].sum()) if long_mask.any() else 0,
        "short_pnl_ticks": float(net_pnls[short_mask].sum()) if short_mask.any() else 0,
        "long_win_rate": float((net_pnls[long_mask] > 0).mean()) if long_mask.any() else 0,
        "short_win_rate": float((net_pnls[short_mask] > 0).mean()) if short_mask.any() else 0,
        "avg_trade_count_per_day": float(len(trades) / n_days),
    }


def format_report(all_results: Dict) -> str:
    """Format results as a human-readable text report."""
    lines = []
    lines.append("=" * 140)
    lines.append("MULTI-HORIZON CONTINUOUS RE-EVALUATION BACKTESTER V2 — Mamba v7 Tiny Smart V3 (March 2026)")
    lines.append("=" * 140)
    lines.append(f"Folds: {FOLDS[0]}-{FOLDS[-1]} (March dates only)")
    lines.append(f"Entry: 1s prediction magnitude in Top-N% confidence tier")
    lines.append(f"P&L: label interpolation at entry (approx, see docstring)")
    lines.append(f"Cost model: passive={COST_PASSIVE:.2f}t, mid={COST_MID:.2f}t, market={COST_MARKET:.2f}t")
    lines.append("")

    for cost_label, cost_val in COST_SCENARIOS.items():
        lines.append(f"\n{'─' * 140}")
        lines.append(f"COST SCENARIO: {cost_label.upper()} ({cost_val:.2f} ticks = ${cost_val * TICK_VALUE:.2f} RT)")
        lines.append(f"{'─' * 140}")

        header = (
            f"{'Strategy':<24} {'Tier':<8} {'Trades':>7} {'GrossPnL':>10} {'NetPnL':>10} "
            f"{'NetUSD':>10} {'WinRate':>8} {'AvgWin':>8} {'AvgLose':>8} "
            f"{'PF':>6} {'Sortino':>8} {'AvgHold':>8} {'AvgEvts':>7} {'L/S':>8} {'Trd/Day':>7}"
        )
        lines.append(header)
        lines.append("─" * 140)

        for strategy_name in EXIT_STRATEGIES:
            for tier_name in CONFIDENCE_TIERS:
                key = f"{strategy_name}__{tier_name}__{cost_label}"
                if key not in all_results:
                    continue
                m = all_results[key]
                if m["n_trades"] == 0:
                    continue

                line = (
                    f"{strategy_name:<24} {tier_name:<8} {m['n_trades']:>7d} "
                    f"{m['total_gross_pnl_ticks']:>10.1f} {m['total_net_pnl_ticks']:>10.1f} "
                    f"${m['net_pnl_usd']:>9.0f} {m['win_rate']:>7.1%} {m['avg_winner_ticks']:>8.2f} "
                    f"{m['avg_loser_ticks']:>8.2f} {m['profit_factor']:>6.2f} "
                    f"{m['sortino']:>8.3f} {m['avg_hold_sec']:>7.1f}s {m['avg_hold_events']:>7.1f} "
                    f"{m['n_long']:>3d}/{m['n_short']:>3d} {m['avg_trade_count_per_day']:>7.1f}"
                )
                lines.append(line)

        lines.append("")

    # ── Top 10 by Sortino at each cost scenario ──
    for cost_label, cost_val in COST_SCENARIOS.items():
        lines.append(f"\n{'=' * 140}")
        lines.append(f"TOP 10 BY SORTINO — {cost_label.upper()} COST ({cost_val:.2f}t)")
        lines.append(f"{'=' * 140}")

        cost_keys = [
            k for k in all_results
            if k.endswith(f"__{cost_label}") and all_results[k]["n_trades"] > 0
        ]
        sorted_keys = sorted(cost_keys, key=lambda k: all_results[k]["sortino"], reverse=True)

        for rank, key in enumerate(sorted_keys[:10], 1):
            m = all_results[key]
            # Parse strategy and tier from key
            parts = key.split("__")
            strat, tier = parts[0], parts[1]
            lines.append(
                f"  #{rank:>2d}: {strat:<24} {tier:<8} | "
                f"Sortino={m['sortino']:>8.3f} | Net={m['total_net_pnl_ticks']:>8.1f}t "
                f"(${m['net_pnl_usd']:>8.0f}) | {m['n_trades']} trades | "
                f"WR={m['win_rate']:.1%} | PF={m['profit_factor']:.2f} | "
                f"AvgHold={m['avg_hold_sec']:.1f}s ({m['avg_hold_events']:.0f}evts) | "
                f"Trd/Day={m['avg_trade_count_per_day']:.1f}"
            )

    # ── Hold time comparison across strategies ──
    lines.append(f"\n{'=' * 140}")
    lines.append("HOLD TIME COMPARISON (Passive cost, Top1%)")
    lines.append(f"{'=' * 140}")

    for strategy_name in EXIT_STRATEGIES:
        key = f"{strategy_name}__Top1%__passive"
        if key not in all_results or all_results[key]["n_trades"] == 0:
            continue
        m = all_results[key]
        lines.append(
            f"  {strategy_name:<24} | AvgHold={m['avg_hold_sec']:>7.1f}s | "
            f"MedianHold={m['median_hold_sec']:>7.1f}s | MaxHold={m['max_hold_sec']:>7.1f}s | "
            f"AvgEvents={m['avg_hold_events']:>5.1f} | Sortino={m['sortino']:.3f}"
        )

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Multi-Horizon Backtester V2 — Loading fold data...")
    print("=" * 80)

    fold_data = []
    for fold_idx in FOLDS:
        print(f"  Loading fold {fold_idx}...", end=" ", flush=True)
        d = load_fold_data(fold_idx)
        if d is not None:
            fold_data.append(d)
            print(f"OK ({d['n_preds']} predictions, date={d['date']})")
        else:
            print("SKIPPED")

    if not fold_data:
        print("ERROR: No fold data loaded!")
        sys.exit(1)

    total_preds = sum(d["n_preds"] for d in fold_data)
    print(f"\nLoaded {len(fold_data)} folds, {total_preds} total predictions")

    # ── Run all strategy x tier combinations, collect raw trades ──
    # First pass: collect trades per (strategy, tier) — cost-independent
    raw_trades: Dict[str, List[Trade]] = {}
    total_combos = len(EXIT_STRATEGIES) * len(CONFIDENCE_TIERS)
    combo_idx = 0

    print(f"\nRunning {total_combos} strategy x tier combinations...")

    for strategy_name, strategy_params in EXIT_STRATEGIES.items():
        for tier_name, tier_pctile in CONFIDENCE_TIERS.items():
            combo_idx += 1
            strat_tier_key = f"{strategy_name}__{tier_name}"
            print(f"\r  [{combo_idx}/{total_combos}] {strat_tier_key:<45}", end="", flush=True)

            all_trades = []
            for data in fold_data:
                trades = run_backtest(data, strategy_name, strategy_params, tier_pctile)
                all_trades.extend(trades)

            raw_trades[strat_tier_key] = all_trades

    print(f"\n\nComputing metrics for {len(raw_trades)} combinations x {len(COST_SCENARIOS)} cost scenarios...")

    # ── Second pass: compute metrics for each cost scenario ──
    all_results = {}
    for strat_tier_key, trades in raw_trades.items():
        for cost_label, cost_val in COST_SCENARIOS.items():
            full_key = f"{strat_tier_key}__{cost_label}"
            all_results[full_key] = compute_metrics(trades, cost_val)

    # ── Print top 10 by Sortino at passive cost ──
    print("\n" + "=" * 80)
    print("TOP 10 BY SORTINO (Passive cost)")
    print("=" * 80)

    passive_keys = [
        k for k in all_results
        if k.endswith("__passive") and all_results[k]["n_trades"] > 0
    ]
    sorted_by_sortino = sorted(passive_keys, key=lambda k: all_results[k]["sortino"], reverse=True)

    for rank, key in enumerate(sorted_by_sortino[:10], 1):
        m = all_results[key]
        parts = key.split("__")
        strat, tier = parts[0], parts[1]
        print(
            f"  #{rank:>2d}: {strat:<24} {tier:<8} | "
            f"Sortino={m['sortino']:.3f} | Net={m['total_net_pnl_ticks']:.1f}t "
            f"(${m['net_pnl_usd']:.0f}) | {m['n_trades']} trades | "
            f"WR={m['win_rate']:.1%} | PF={m['profit_factor']:.2f} | "
            f"AvgHold={m['avg_hold_sec']:.1f}s ({m['avg_hold_events']:.0f}evts)"
        )

    # ── Save results JSON ──
    results_path = OUTPUT_DIR / "backtest_multi_horizon_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Save formatted report ──
    report = format_report(all_results)
    report_path = OUTPUT_DIR / "backtest_multi_horizon_v2_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
