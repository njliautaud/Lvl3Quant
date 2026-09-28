#!/usr/bin/env python3
"""
Passive Exit Fill Rate Simulator v1
=====================================
Estimates realistic passive exit fill rates for short strategy.

Core question: After a passive SHORT entry fill (sell limit at ask), if we place
a limit BUY at the bid to exit passively, how often do we actually get filled
within a reasonable hold window?

Methodology:
- For each top-N% short signal, assume passive entry at ask (sell limit filled).
- After entry, track mid price path at 1s resolution by chaining labels_1s.
  At entry event e0: labels_1s[e0] = mid change over next 1s (exact).
  Find event e1 at t+1s, labels_1s[e1] = mid change from t+1s to t+2s. Chain forward.
- Also use direct horizon labels (5s, 10s, 30s) as ground-truth checkpoints.
- For each hold window, track MINIMUM mid reached (most favorable for short exit).
- A passive exit fill happens when mid drops enough from entry.

Exit Scenarios:
  A: "Full spread" — mid drops >= 0.5 ticks (passive buy at bid fills).
     Gross = +1.0 tick (sell at ask, buy at bid). Net = +1.0 - 0.376 = +0.624.
  B: "Conservative FIFO" — mid drops >= 1.0 tick (deeper in queue, need more movement).
     Same gross but requires more price movement to actually fill.
  C: "Ultra conservative" — mid drops >= 1.5 ticks (very back of queue).
     Hardest to fill but most realistic for large queue.

Unfilled exits must market-out: buy at ask = entry_price. Gross = 0. Net = -0.376.

Cost model:
  Commission: 0.376 ticks RT ($4.70 / $12.50)
  Passive entry + passive exit: net = +0.624 ticks (earn spread, pay commission)
  Passive entry + market exit: net = signal_pnl - 0.376 ticks

Author: Claude (Head of Quant)
Date: 2026-05-23
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output/cnn_mamba_v2_all_oot"
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OUT_DIR = ROOT / "output/passive_exit_fill_sim_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 3000
STRIDE = 250
COMMISSION_TICKS_RT = 0.376
ES_TICK_VALUE = 12.50

# Exit fill thresholds: how much mid must drop for passive exit fill (ticks)
# After selling at ask, mid = ask - 0.5. To buy at bid = ask - 1 = mid - 0.5:
# Need mid to drop 0.5 ticks. More conservative = need more drop for FIFO queue.
FILL_THRESHOLDS = {
    "optimistic_0.5": 0.5,    # Mid drops 0.5 tick → at bid level
    "moderate_1.0": 1.0,      # Mid drops 1.0 tick → trades through bid (FIFO fill)
    "conservative_1.5": 1.5,  # Mid drops 1.5 ticks → deep FIFO fill
    "very_conserv_2.0": 2.0,  # Mid drops 2.0 ticks → very conservative
}

# Hold windows (seconds)
HOLD_WINDOWS_SEC = [5, 10, 30, 60, 120]

# Confidence thresholds
CONFIDENCE_PCTS = [1, 2, 5, 10, 20]

# P&L constants
PASSIVE_EXIT_NET = 1.0 - COMMISSION_TICKS_RT  # +0.624 ticks (spread - commission)
# For market exit: P&L depends on where mid is. sell at ask, buy at ask later:
# P&L = -(mid_change) - commission. If mid unchanged: net = -0.376


def find_overlap_dates():
    """Find dates where both predictions and MBO events exist."""
    pred_dates = {f[:8] for f in os.listdir(PRED_DIR)
                  if f.endswith("_predictions.npz") and f[0] == "2"}
    mbo_dates = {f[:8] for f in os.listdir(MBO_DIR)
                 if f.endswith("_mbo_events.npz")}
    return sorted(pred_dates & mbo_dates)


def load_day(date_str: str):
    """Load data for one day. Returns predictions, all label arrays, timestamps."""
    pred_data = np.load(PRED_DIR / f"{date_str}_predictions.npz", allow_pickle=True)
    mbo_data = np.load(MBO_DIR / f"{date_str}_mbo_events.npz", mmap_mode="r")

    predictions = pred_data["predictions"]  # (N, 3)
    labels_1s = np.array(mbo_data["labels_1s"])
    labels_5s = np.array(mbo_data["labels_5s"])
    labels_10s = np.array(mbo_data["labels_10s"])
    labels_30s = np.array(mbo_data["labels_30s"])
    timestamps = np.array(mbo_data["timestamps"])

    # Trim predictions if needed
    n_events = len(labels_1s)
    expected = (predictions.shape[0] - 1) * STRIDE + WINDOW_SIZE
    if expected > n_events:
        max_preds = (n_events - WINDOW_SIZE) // STRIDE + 1
        predictions = predictions[:max_preds]

    return predictions, labels_1s, labels_5s, labels_10s, labels_30s, timestamps


def build_mid_path_1s(entry_event: int, labels_1s: np.ndarray,
                      timestamps: np.ndarray, max_seconds: int = 130) -> np.ndarray:
    """Build mid price path at ~1s resolution from entry event.

    Chains labels_1s forward: each step adds the 1s label at the current event.
    Returns array of (time_offset_s, cumulative_mid_change) pairs.
    """
    path = [(0.0, 0.0)]  # (time_s, cum_mid_change)
    cum_mid = 0.0
    current_event = entry_event
    t_entry = timestamps[entry_event]
    n_events = len(timestamps)

    for step in range(max_seconds + 5):  # up to max_seconds 1s steps
        if current_event >= len(labels_1s):
            break

        label = labels_1s[current_event]
        if np.isnan(label):
            # Skip but try to advance
            t_target = timestamps[current_event] + 1_000_000_000
            next_event = np.searchsorted(timestamps, t_target)
            if next_event >= n_events or next_event == current_event:
                break
            current_event = next_event
            continue

        cum_mid += label

        # Find event ~1s later
        t_target = timestamps[current_event] + 1_000_000_000
        next_event = np.searchsorted(timestamps, t_target)
        if next_event >= n_events or next_event == current_event:
            break

        dt_from_entry = (timestamps[next_event] - t_entry) / 1e9
        if dt_from_entry < 0 or dt_from_entry > max_seconds + 10:
            break

        path.append((dt_from_entry, cum_mid))
        current_event = next_event

    return np.array(path) if len(path) > 1 else np.array([[0.0, 0.0]])


def add_horizon_checkpoints(entry_event: int, labels_1s, labels_5s,
                            labels_10s, labels_30s) -> list:
    """Get exact mid-change checkpoints from direct horizon labels."""
    checkpoints = []
    for horizon_s, labels in [(1, labels_1s), (5, labels_5s),
                               (10, labels_10s), (30, labels_30s)]:
        if entry_event < len(labels):
            val = labels[entry_event]
            if not np.isnan(val):
                checkpoints.append((float(horizon_s), float(val)))
    return checkpoints


def simulate_day(predictions, labels_1s, labels_5s, labels_10s, labels_30s,
                 timestamps, confidence_pct):
    """Simulate passive exit fill rates for one day at given confidence level.

    Returns list of trade dicts with fill info per threshold/window.
    """
    preds_1s = predictions[:, 0]
    n_preds = len(preds_1s)
    decision_indices = np.arange(n_preds) * STRIDE + (WINDOW_SIZE - 1)

    max_event = len(labels_1s) - 1
    valid = decision_indices <= max_event
    decision_indices = decision_indices[valid]
    preds_1s = preds_1s[:len(decision_indices)]

    # Threshold for short signals
    threshold = np.nanpercentile(preds_1s, confidence_pct)
    if threshold >= 0:
        return []

    signal_mask = preds_1s <= threshold
    signal_indices = np.where(signal_mask)[0]

    trades = []
    cooldown_until = -1

    for sig_idx in signal_indices:
        if sig_idx <= cooldown_until:
            continue

        d_idx = decision_indices[sig_idx]
        label_1s = labels_1s[d_idx]
        if np.isnan(label_1s):
            continue

        # Build mid path at 1s resolution
        path = build_mid_path_1s(d_idx, labels_1s, timestamps, max_seconds=130)

        # Also get exact horizon checkpoints
        checkpoints = add_horizon_checkpoints(d_idx, labels_1s, labels_5s,
                                               labels_10s, labels_30s)

        # For 60s and 120s, chain from 30s checkpoint
        # Find event at ~30s from entry for additional chaining
        t_entry = timestamps[d_idx]
        t_30s = t_entry + 30_000_000_000
        event_30s = np.searchsorted(timestamps, t_30s)
        if event_30s < len(labels_30s) and not np.isnan(labels_30s[d_idx]):
            mid_at_30s = labels_30s[d_idx]
            if event_30s < len(labels_30s) and not np.isnan(labels_30s[event_30s]):
                checkpoints.append((60.0, mid_at_30s + labels_30s[event_30s]))

            # For 120s: chain 30s → 60s → 90s → 120s
            t_60s = t_entry + 60_000_000_000
            event_60s = np.searchsorted(timestamps, t_60s)
            t_90s = t_entry + 90_000_000_000
            event_90s = np.searchsorted(timestamps, t_90s)

            mid_at_60 = None
            mid_at_90 = None
            if event_60s < len(labels_30s) and not np.isnan(labels_30s[event_60s]):
                if event_30s < len(labels_30s) and not np.isnan(labels_30s[event_30s]):
                    mid_at_60 = mid_at_30s + labels_30s[event_30s]
                    checkpoints.append((90.0, mid_at_60 + labels_30s[event_60s]))
            if event_90s < len(labels_30s) and not np.isnan(labels_30s[event_90s]):
                if mid_at_60 is not None:
                    mid_at_90 = mid_at_60 + labels_30s[event_60s]
                    checkpoints.append((120.0, mid_at_90 + labels_30s[event_90s]))

        # Combine path and checkpoints
        all_points = list(zip(path[:, 0], path[:, 1])) + checkpoints
        all_points.sort(key=lambda x: x[0])

        # For each hold window, find MINIMUM mid change within that window
        # (most negative = most favorable for short exit)
        trade = {
            "pred_value": float(preds_1s[sig_idx]),
            "label_1s": float(label_1s),
        }

        for window_s in HOLD_WINDOWS_SEC:
            points_in_window = [p[1] for p in all_points if 0 < p[0] <= window_s]
            min_mid = min(points_in_window) if points_in_window else 0.0
            # Also track the final mid change at window end
            points_near_end = [p for p in all_points if abs(p[0] - window_s) <= 2]
            final_mid = points_near_end[-1][1] if points_near_end else 0.0

            trade[f"min_mid_{window_s}s"] = float(min_mid)
            trade[f"final_mid_{window_s}s"] = float(final_mid)

        trades.append(trade)
        cooldown_until = sig_idx + 4  # ~1s cooldown

    return trades


def compute_metrics(trades: list, n_days: int) -> dict:
    """Compute fill rates, blended P&L, Sharpe for all scenario/window combos."""
    n_trades = len(trades)
    if n_trades == 0:
        return {}

    results = {}

    for thresh_name, thresh_val in FILL_THRESHOLDS.items():
        results[thresh_name] = {}

        for window_s in HOLD_WINDOWS_SEC:
            min_key = f"min_mid_{window_s}s"
            final_key = f"final_mid_{window_s}s"

            fills = []
            blended_pnls = []
            daily_pnl = {}  # date_proxy → pnl sum for Sharpe

            for i, trade in enumerate(trades):
                min_mid = trade.get(min_key, 0.0)

                # Fill if min mid change <= -threshold (mid dropped enough)
                filled = min_mid <= -thresh_val

                if filled:
                    # Passive exit: earn spread minus commission
                    pnl = PASSIVE_EXIT_NET  # +0.624 ticks
                else:
                    # Market exit at end of hold window
                    # sell at ask (= mid + 0.5), buy at ask (= current_mid + 0.5)
                    # P&L = (entry_mid + 0.5) - (current_mid + 0.5) = -(current_mid - entry_mid)
                    #      = -final_mid_change
                    final_mid = trade.get(final_key, 0.0)
                    pnl = -final_mid - COMMISSION_TICKS_RT

                fills.append(filled)
                blended_pnls.append(pnl)

                # Group by approximate day for Sharpe calculation
                day_proxy = i // max(n_trades // n_days, 1)
                daily_pnl.setdefault(day_proxy, []).append(pnl)

            fill_rate = sum(fills) / n_trades
            avg_blended = np.mean(blended_pnls)

            # Daily Sharpe
            daily_sums = [sum(v) for v in daily_pnl.values()]
            if len(daily_sums) > 1:
                dm = np.mean(daily_sums)
                ds = np.std(daily_sums, ddof=1)
                sharpe = dm / ds * np.sqrt(252) if ds > 1e-9 else 0
                neg = [d for d in daily_sums if d < 0]
                dd = np.std(neg, ddof=1) if len(neg) > 1 else ds
                sortino = dm / dd * np.sqrt(252) if dd > 1e-9 else 0
            else:
                sharpe = sortino = 0.0

            # Win rate and profit factor
            wr = sum(1 for p in blended_pnls if p > 0) / n_trades
            wins = sum(p for p in blended_pnls if p > 0)
            losses = abs(sum(p for p in blended_pnls if p < 0))
            pf = wins / losses if losses > 0 else float("inf")

            # Break-even fill rate
            # blended = fr * 0.624 + (1-fr) * avg_market_pnl = 0
            # fr = -avg_market_pnl / (0.624 - avg_market_pnl)
            unfilled_pnls = [blended_pnls[i] for i in range(n_trades) if not fills[i]]
            avg_market_pnl = np.mean(unfilled_pnls) if unfilled_pnls else 0
            if abs(PASSIVE_EXIT_NET - avg_market_pnl) > 1e-9:
                be_fr = -avg_market_pnl / (PASSIVE_EXIT_NET - avg_market_pnl)
            else:
                be_fr = float("inf")

            results[thresh_name][window_s] = {
                "fill_rate": float(fill_rate),
                "n_filled": int(sum(fills)),
                "n_trades": n_trades,
                "avg_blended_pnl": float(avg_blended),
                "avg_market_exit_pnl": float(avg_market_pnl),
                "total_ticks": float(sum(blended_pnls)),
                "total_dollars": float(sum(blended_pnls) * ES_TICK_VALUE),
                "sharpe": float(sharpe),
                "sortino": float(sortino),
                "win_rate": float(wr),
                "profit_factor": float(pf) if pf < 1000 else float("inf"),
                "breakeven_fill_rate": float(be_fr),
            }

    return results


def print_results_table(all_results: dict):
    """Print comprehensive results tables."""
    print(f"\n{'='*140}")
    print(f"PASSIVE EXIT FILL RATE SIMULATION RESULTS")
    print(f"{'='*140}")

    for thresh_name, thresh_val in FILL_THRESHOLDS.items():
        print(f"\n{'─'*140}")
        print(f"Fill Threshold: {thresh_name} (mid must drop >= {thresh_val} ticks for fill)")
        print(f"{'─'*140}")

        header = (f"{'Conf%':>5} {'Window':>6} │ {'FillRate':>8} {'Filled':>6}/{'':<5} │ "
                  f"{'BlendPnL':>8} {'TotalTk':>8} {'Total$':>9} │ "
                  f"{'Sharpe':>7} {'Sortino':>7} {'WR':>5} {'PF':>5} │ {'BE_FR':>6}")
        print(header)
        print(f"{'─'*140}")

        for conf_pct in CONFIDENCE_PCTS:
            key = f"top_{conf_pct}pct"
            if key not in all_results or thresh_name not in all_results[key]:
                continue

            for w in HOLD_WINDOWS_SEC:
                if w not in all_results[key][thresh_name]:
                    continue

                m = all_results[key][thresh_name][w]
                pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] < 100 else "inf"
                be_str = f"{m['breakeven_fill_rate']:.0%}" if 0 < m['breakeven_fill_rate'] < 5 else "N/A"

                print(f"{conf_pct:5d} {w:5d}s │ {m['fill_rate']:7.1%} {m['n_filled']:6d}/{m['n_trades']:<5d} │ "
                      f"{m['avg_blended_pnl']:+8.3f} {m['total_ticks']:+8.0f} ${m['total_dollars']:+8.0f} │ "
                      f"{m['sharpe']:7.1f} {m['sortino']:7.1f} {m['win_rate']:5.1%} {pf_str:>5} │ {be_str:>6}")

            print()

    # ── Summary: best combos ──
    print(f"\n{'='*140}")
    print(f"BEST COMBOS: Highest blended P&L per trade (positive only)")
    print(f"{'='*140}")
    print(f"{'Conf%':>5} {'Threshold':>20} {'Window':>6} │ {'FillRate':>8} {'BlendPnL':>8} "
          f"{'Sharpe':>7} {'$/trade':>8} │ {'Trades':>6} {'Total$':>9}")
    print(f"{'─'*100}")

    combos = []
    for conf_pct in CONFIDENCE_PCTS:
        key = f"top_{conf_pct}pct"
        if key not in all_results:
            continue
        for thresh_name in FILL_THRESHOLDS:
            if thresh_name not in all_results[key]:
                continue
            for w in HOLD_WINDOWS_SEC:
                if w not in all_results[key][thresh_name]:
                    continue
                m = all_results[key][thresh_name][w]
                combos.append((conf_pct, thresh_name, w, m))

    combos.sort(key=lambda x: x[3]["avg_blended_pnl"], reverse=True)
    for conf_pct, thresh_name, w, m in combos[:20]:
        if m["avg_blended_pnl"] <= 0:
            break
        dollar_per_trade = m["avg_blended_pnl"] * ES_TICK_VALUE
        print(f"{conf_pct:5d} {thresh_name:>20} {w:5d}s │ {m['fill_rate']:7.1%} "
              f"{m['avg_blended_pnl']:+8.3f} {m['sharpe']:7.1f} ${dollar_per_trade:+7.2f} │ "
              f"{m['n_trades']:6d} ${m['total_dollars']:+8.0f}")


def print_key_findings(all_results: dict):
    """Print the most important findings."""
    print(f"\n{'='*100}")
    print(f"KEY FINDINGS")
    print(f"{'='*100}")

    # For each confidence level, show the moderate threshold at 30s
    print("\n  MODERATE fill requirement (mid drops 1.0 tick = trades through bid):")
    for conf_pct in CONFIDENCE_PCTS:
        key = f"top_{conf_pct}pct"
        if key not in all_results or "moderate_1.0" not in all_results[key]:
            continue
        for w in [10, 30, 60]:
            if w not in all_results[key]["moderate_1.0"]:
                continue
            m = all_results[key]["moderate_1.0"][w]
            status = "PROFITABLE" if m["avg_blended_pnl"] > 0 else "UNPROFITABLE"
            print(f"    Top {conf_pct:2d}% shorts, {w:3d}s hold: fill {m['fill_rate']:.0%}, "
                  f"blended {m['avg_blended_pnl']:+.3f} ticks ({m['avg_blended_pnl']*ES_TICK_VALUE:+.2f}$/trade), "
                  f"Sharpe {m['sharpe']:.1f} [{status}]")
        print()

    # Show the break-even fill rate analysis
    print("\n  BREAK-EVEN FILL RATES (what fill rate makes blended P&L = 0):")
    for conf_pct in [5, 10]:
        key = f"top_{conf_pct}pct"
        if key not in all_results:
            continue
        print(f"    Top {conf_pct}% shorts:")
        for thresh_name in FILL_THRESHOLDS:
            if thresh_name not in all_results[key]:
                continue
            for w in [30, 60]:
                if w not in all_results[key][thresh_name]:
                    continue
                m = all_results[key][thresh_name][w]
                be = m["breakeven_fill_rate"]
                actual = m["fill_rate"]
                if 0 < be < 5:
                    margin = actual - be
                    status = f"MARGIN: {margin:+.0%}" if margin > 0 else f"SHORT BY: {margin:+.0%}"
                    print(f"      {thresh_name:>20}, {w}s: actual={actual:.0%}, breakeven={be:.0%} [{status}]")

    # Comparison: passive vs market exit
    print(f"\n  VALUE OF PASSIVE EXIT (vs always market exit):")
    print(f"  {'Conf%':>5} {'Window':>6} │ {'Fill%':>6} {'Blended':>8} {'AlwaysMkt':>9} │ {'Δ ticks':>8} {'Δ $/trade':>9}")
    for conf_pct in CONFIDENCE_PCTS:
        key = f"top_{conf_pct}pct"
        if key not in all_results or "moderate_1.0" not in all_results[key]:
            continue
        for w in [30, 60]:
            if w not in all_results[key]["moderate_1.0"]:
                continue
            m = all_results[key]["moderate_1.0"][w]
            # "Always market exit" = every trade exits at market
            # That means P&L = -final_mid - commission for every trade
            # Our blended = fill_rate * 0.624 + (1-fill_rate) * market_exit
            # The improvement is: blended - always_market
            # always_market for the average trade = avg(m["avg_market_exit_pnl"]) approximately
            # But more precisely, the always-market PnL = what happens if we just use
            # the signal with market exit. Let's estimate:
            # With market exit on ALL trades: avg_pnl ≈ avg(-final_mid) - 0.376
            # This is what m["avg_market_exit_pnl"] already represents for unfilled trades.
            # For simplicity, the improvement from passive exit is the fill_rate * (0.624 - market_pnl)
            fr = m["fill_rate"]
            mkt = m["avg_market_exit_pnl"]
            improvement = fr * (PASSIVE_EXIT_NET - mkt)
            print(f"  {conf_pct:5d} {w:5d}s │ {fr:5.0%} {m['avg_blended_pnl']:+8.3f} "
                  f"{mkt:+9.3f} │ {improvement:+8.3f} ${improvement*ES_TICK_VALUE:+8.2f}")


def run_simulation():
    """Run the full simulation."""
    dates = find_overlap_dates()
    print(f"Found {len(dates)} overlap dates: {dates[0]} to {dates[-1]}")

    valid_dates = []
    for d in dates:
        if (PRED_DIR / f"{d}_predictions.npz").exists() and \
           (MBO_DIR / f"{d}_mbo_events.npz").exists():
            valid_dates.append(d)
    print(f"Validated {len(valid_dates)} dates")

    print(f"\n{'='*100}")
    print(f"PASSIVE EXIT FILL RATE SIMULATION")
    print(f"Commission: {COMMISSION_TICKS_RT} ticks RT")
    print(f"Passive exit net: {PASSIVE_EXIT_NET:+.3f} ticks/trade (spread - commission)")
    print(f"Fill thresholds: {list(FILL_THRESHOLDS.keys())}")
    print(f"Hold windows: {HOLD_WINDOWS_SEC}")
    print(f"Confidence levels: {CONFIDENCE_PCTS}")
    total_combos = len(FILL_THRESHOLDS) * len(HOLD_WINDOWS_SEC) * len(CONFIDENCE_PCTS)
    print(f"Total scenario combos: {total_combos}")
    print(f"{'='*100}\n")

    all_results = {}

    for conf_pct in CONFIDENCE_PCTS:
        print(f"\n── Top {conf_pct}% short signals ──")
        t_start = time.time()

        all_trades = []

        for day_idx, date_str in enumerate(sorted(valid_dates)):
            try:
                data = load_day(date_str)
            except Exception as e:
                print(f"  WARN: {date_str}: {e}")
                continue

            predictions, labels_1s, labels_5s, labels_10s, labels_30s, timestamps = data

            day_trades = simulate_day(predictions, labels_1s, labels_5s,
                                       labels_10s, labels_30s, timestamps, conf_pct)
            all_trades.extend(day_trades)

            if day_idx % 10 == 0 or day_idx == len(valid_dates) - 1:
                print(f"  Day {day_idx+1}/{len(valid_dates)}: {date_str} — "
                      f"{len(day_trades)} trades (cum: {len(all_trades)})")

            del data, predictions, labels_1s, labels_5s, labels_10s, labels_30s, timestamps

        elapsed = time.time() - t_start
        print(f"  Total: {len(all_trades)} trades in {elapsed:.1f}s")

        if all_trades:
            metrics = compute_metrics(all_trades, len(valid_dates))
            all_results[f"top_{conf_pct}pct"] = metrics

    # ── Print results ────────────────────────────────────────────────────
    print_results_table(all_results)
    print_key_findings(all_results)

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "run_date": "2026-05-23",
        "n_dates": len(valid_dates),
        "date_range": f"{valid_dates[0]} to {valid_dates[-1]}",
        "methodology": {
            "entry": "Passive sell limit at ask (assumed filled for qualifying short signals)",
            "exit": "Limit buy at bid to exit passively, or market buy at ask if unfilled",
            "mid_path": "1s-resolution chaining of labels_1s + direct horizon checkpoints",
            "fill_condition": "Min mid change within hold window <= -threshold",
            "passive_exit_net": PASSIVE_EXIT_NET,
            "commission_rt": COMMISSION_TICKS_RT,
            "fill_thresholds": {k: v for k, v in FILL_THRESHOLDS.items()},
            "hold_windows_sec": HOLD_WINDOWS_SEC,
        },
        "results": {},
    }

    # Convert results for JSON serialization
    for conf_key, conf_data in all_results.items():
        output["results"][conf_key] = {}
        for thresh_key, thresh_data in conf_data.items():
            output["results"][conf_key][thresh_key] = {}
            for w, metrics in thresh_data.items():
                # Convert inf to string for JSON
                clean = {}
                for k, v in metrics.items():
                    if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                        clean[k] = str(v)
                    else:
                        clean[k] = v
                output["results"][conf_key][thresh_key][str(w)] = clean

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    run_simulation()
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
