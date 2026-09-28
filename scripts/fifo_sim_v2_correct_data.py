#!/usr/bin/env python3
"""
FIFO Cancel/Reprice Simulator v1
=================================
Simulates passive limit order placement with cancel/reprice logic
for top-confidence CNN-Mamba v2 SHORT signals on ES futures.

Core hypothesis: canceling unfilled passive orders when mid moves against
signal within a timeout, then repricing up to N times, should avoid
adverse-selection losses and improve net ticks/trade.

Methodology:
- At each prediction stride (~250ms), check for high-confidence short signals
- Place sell limit at ask (back of FIFO queue)
- Track mid price movement using label-differencing between strides:
    mid_change(k → k+n) ≈ labels_1s[d_k] - labels_1s[d_{k+n}]
- Cancel if mid moves UP (against short) by threshold within timeout
- Reprice at new ask up to max_reprices times
- If filled (mid doesn't move against within timeout → assume fill at ask):
    P&L = -(labels_1s at fill point) - costs

Cost model (canonical HC values):
  Commission: 0.376 ticks RT ($4.70 / $12.50)
  Passive fill: 0 spread crossing (at ask for shorts)
  Exit: 1.0 tick spread crossing (market order exit)
  Total cost per completed trade: 1.376 ticks

Author: Claude (Head of Quant)
Date: 2026-05-23
"""

import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output/cnn_mamba_v2_bulk_oot_v2"
MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
OUT_DIR = ROOT / "output/fifo_sim_v2_correct_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 3000
STRIDE = 250
COMMISSION_TICKS_RT = 0.376       # $4.70 / $12.50

# Cost scenarios:
# Passive entry (sell at ask) + market exit (buy at ask):
#   PnL = -labels_1s, cost = commission only
#   Reason: both sides at ask price → spread cancels out
# Passive entry (sell at ask) + passive exit (buy at bid):
#   PnL = -labels_1s + 1.0 (earn full spread), cost = commission
COST_MARKET_EXIT = COMMISSION_TICKS_RT          # 0.376 ticks
COST_PASSIVE_EXIT = COMMISSION_TICKS_RT - 1.0   # -0.624 ticks (you EARN spread on both sides)

# ── Feature columns (from empirical_fill_model_v1.py) ────────────────────────
FEATURE_NAMES = [
    "time_delta_log", "event_type_id", "side_id", "price_rel_ticks", "qty_log",
    "spread_ticks", "cancel_side_asym_50", "rolling_ofi_500", "event_density_20",
    "price_mom_10", "qty_price_mom_50", "price_sign_mom_200",
    "event_type_entropy_200", "fill_add_restoration_100", "spread_velocity_50",
    "queue_replenishment", "mom_divergence", "ofi_x_spread",
    "vol_weighted_pmom", "buy_sell_intensity", "realized_volatility",
    "sweep_intensity", "ofi_short_100", "ofi_long_2000", "ofi_acceleration",
]


def find_overlap_dates():
    """Find dates where both predictions and MBO events exist."""
    pred_dates = set()
    for f in os.listdir(PRED_DIR):
        if f.endswith("_predictions.npz") and f[0] == "2":
            pred_dates.add(f[:8])

    mbo_dates = set()
    for f in os.listdir(MBO_DIR):
        if f.endswith("_mbo_events.npz"):
            mbo_dates.add(f[:8])

    overlap = sorted(pred_dates & mbo_dates)
    return overlap


def load_day(date_str: str):
    """Load predictions and MBO event data for one day.

    Returns:
        predictions: (N, 3) array — [1s, 5s, 10s] horizon predictions
        labels_1s: (M,) array — 1s-ahead mid price change in ticks at each event
        timestamps: (M,) array — nanosecond timestamps for each event
        events: (M, 25) array — feature matrix
    """
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"

    pred_data = np.load(pred_path, allow_pickle=True)
    mbo_data = np.load(mbo_path)

    predictions = pred_data["predictions"]  # (N, 3)
    labels_1s = mbo_data["labels_1s"]       # (M,)
    timestamps = mbo_data["timestamps"]     # (M,)
    events = mbo_data["events"]             # (M, 25)

    return predictions, labels_1s, timestamps, events


def compute_decision_indices(n_predictions: int) -> np.ndarray:
    """Compute event indices for each prediction's decision point."""
    return np.arange(n_predictions) * STRIDE + (WINDOW_SIZE - 1)


def compute_confidence_threshold(predictions_1s: np.ndarray, pct: float) -> float:
    """Compute the prediction threshold for top pct% of short signals.

    Short signals have NEGATIVE predictions (predicting mid goes down).
    Top pct% means the most negative predictions.
    """
    # For shorts: lower (more negative) = more confident
    threshold = np.nanpercentile(predictions_1s, pct)
    return threshold


def simulate_day(
    predictions: np.ndarray,
    labels_1s: np.ndarray,
    timestamps: np.ndarray,
    events,  # unused, kept for API compat
    confidence_pct: float,
    cancel_timeout_ms: float,
    mid_move_threshold: float,
    max_reprices: int,
) -> dict:
    """Simulate one day of cancel/reprice trading.

    Args:
        predictions: (N, 3) prediction array
        labels_1s: (M,) labels array at event level
        timestamps: (M,) nanosecond timestamps
        events: unused (kept for compatibility)
        confidence_pct: top N% threshold (e.g., 5 for top 5%)
        cancel_timeout_ms: ms to wait before checking cancel condition
        mid_move_threshold: ticks of adverse mid move to trigger cancel
        max_reprices: max number of reprices after cancel (0 = cancel & abandon)

    Returns:
        dict with trade-level results
    """
    preds_1s = predictions[:, 0]  # 1s horizon predictions
    n_preds = len(preds_1s)
    decision_indices = compute_decision_indices(n_preds)

    # Compute confidence threshold for short signals
    # Short signals: negative predictions (predicting mid goes down)
    threshold = compute_confidence_threshold(preds_1s, confidence_pct)
    if threshold >= 0:
        # Not enough negative signals at this threshold
        return {"trades": [], "n_signals": 0, "threshold": float(threshold)}

    # Find short signal indices
    signal_mask = preds_1s <= threshold
    signal_indices = np.where(signal_mask)[0]

    # Pre-compute timestamps at decision points
    pred_timestamps_ns = timestamps[decision_indices]

    # Pre-compute labels at decision points for mid-change estimation
    labels_at_decisions = labels_1s[decision_indices]

    trades = []
    active_order = None  # Track current order state
    cooldown_until = -1  # Prevent overlapping signals

    for sig_idx in signal_indices:
        d_idx = decision_indices[sig_idx]

        # Skip if in cooldown (previous trade still in hold period)
        if sig_idx <= cooldown_until:
            continue

        # Skip if near end of day (need room for hold period)
        # 1s horizon → need ~4 strides after signal (4 * ~250ms ≈ 1s)
        if sig_idx + 8 >= n_preds:
            continue

        # Skip if label is NaN at decision point
        if np.isnan(labels_at_decisions[sig_idx]):
            continue

        pred_value = float(preds_1s[sig_idx])
        signal_ts = pred_timestamps_ns[sig_idx]

        # ── Order placement and cancel/reprice loop ──
        filled = False
        fill_stride = sig_idx  # stride index where fill occurs
        n_reprices_done = 0
        order_stride = sig_idx  # current order placement stride

        while True:
            # Check strides forward for cancel condition
            # Convert cancel_timeout_ms to approximate number of strides
            # Stride ≈ 250ms median, but use actual timestamps for accuracy

            cancel_triggered = False
            fill_assumed = False

            # Look ahead from order_stride to find cancel or fill
            for look_ahead in range(1, min(20, n_preds - order_stride)):
                check_stride = order_stride + look_ahead
                if check_stride >= n_preds:
                    break

                check_ts = pred_timestamps_ns[check_stride]
                elapsed_ms = (check_ts - pred_timestamps_ns[order_stride]) / 1e6

                # Skip if timestamps are weird (session breaks)
                if elapsed_ms > 30000 or elapsed_ms < 0:
                    break

                if np.isnan(labels_at_decisions[check_stride]):
                    continue

                # Estimate mid price change from order placement to now
                # mid_change ≈ labels_1s[d_order] - labels_1s[d_check]
                mid_change = labels_at_decisions[order_stride] - labels_at_decisions[check_stride]

                # For SHORT signal: adverse move = mid going UP (positive mid_change)
                # mid_change > 0 means price went up from order to check → bad for short

                if elapsed_ms >= cancel_timeout_ms:
                    if mid_change > mid_move_threshold:
                        # Mid moved against us → cancel
                        cancel_triggered = True
                        cancel_stride = check_stride
                        break
                    else:
                        # Mid didn't move against us significantly → assume fill
                        fill_assumed = True
                        fill_stride = check_stride
                        break

            if fill_assumed:
                filled = True
                break
            elif cancel_triggered:
                if n_reprices_done < max_reprices:
                    # Reprice: move order to new ask (at cancel_stride)
                    n_reprices_done += 1
                    order_stride = cancel_stride
                    # Continue the while loop to check the repriced order
                    continue
                else:
                    # Max reprices exhausted → abandon
                    filled = False
                    break
            else:
                # Ran out of look-ahead without triggering cancel or fill
                # Assume fill if we got through without adverse move
                if order_stride + 1 < n_preds:
                    filled = True
                    fill_stride = min(order_stride + 2, n_preds - 1)
                break

        if filled:
            fill_label = labels_at_decisions[fill_stride]
            if np.isnan(fill_label):
                continue

            # Gross P&L: short profits from mid going down
            # labels_1s > 0 means mid went UP → loss for short
            # labels_1s < 0 means mid went DOWN → profit for short
            gross_ticks = -fill_label

            # Net P&L: market exit (conservative) and passive exit (optimistic)
            net_mkt = gross_ticks - COST_MARKET_EXIT    # exit at ask
            net_pas = gross_ticks - COST_PASSIVE_EXIT   # exit at bid (earn spread)

            trades.append({
                "signal_stride": int(sig_idx),
                "fill_stride": int(fill_stride),
                "n_reprices": n_reprices_done,
                "pred_value": pred_value,
                "gross_ticks": float(gross_ticks),
                "net_mkt_ticks": float(net_mkt),
                "net_pas_ticks": float(net_pas),
                "fill_label": float(fill_label),
            })

            # Cooldown: don't place new orders for ~1s (4 strides) after fill
            cooldown_until = fill_stride + 4
        else:
            # Order canceled/abandoned — no trade, no cost
            cooldown_until = max(order_stride + 2, sig_idx + 2)

    return {
        "trades": trades,
        "n_signals": int(len(signal_indices)),
        "threshold": float(threshold),
    }


def simulate_day_baseline(
    predictions: np.ndarray,
    labels_1s: np.ndarray,
    timestamps: np.ndarray,
    confidence_pct: float,
) -> dict:
    """Baseline: passive limit order WITHOUT cancel/reprice.

    Every qualifying signal gets filled (naive assumption that passive
    limits at ask always fill). This is the comparison point.
    """
    preds_1s = predictions[:, 0]
    n_preds = len(preds_1s)
    decision_indices = compute_decision_indices(n_preds)
    labels_at_decisions = labels_1s[decision_indices]

    threshold = compute_confidence_threshold(preds_1s, confidence_pct)
    if threshold >= 0:
        return {"trades": [], "n_signals": 0, "threshold": float(threshold)}

    signal_mask = preds_1s <= threshold
    signal_indices = np.where(signal_mask)[0]

    trades = []
    cooldown_until = -1

    for sig_idx in signal_indices:
        if sig_idx <= cooldown_until:
            continue
        if sig_idx + 4 >= n_preds:
            continue

        label = labels_at_decisions[sig_idx]
        if np.isnan(label):
            continue

        gross_ticks = -label
        net_mkt = gross_ticks - COST_MARKET_EXIT
        net_pas = gross_ticks - COST_PASSIVE_EXIT

        trades.append({
            "signal_stride": int(sig_idx),
            "pred_value": float(preds_1s[sig_idx]),
            "gross_ticks": float(gross_ticks),
            "net_mkt_ticks": float(net_mkt),
            "net_pas_ticks": float(net_pas),
        })

        cooldown_until = sig_idx + 4

    return {
        "trades": trades,
        "n_signals": int(len(signal_indices)),
        "threshold": float(threshold),
    }


def compute_metrics(all_day_results: list, dates: list) -> dict:
    """Compute aggregate metrics from multi-day results.

    Reports both market-exit and passive-exit scenarios.
    """
    all_trades = []
    daily_pnl_mkt = []
    daily_pnl_pas = []

    for day_result, date_str in zip(all_day_results, dates):
        trades = day_result["trades"]
        all_trades.extend(trades)
        day_mkt = sum(t["net_mkt_ticks"] for t in trades)
        day_pas = sum(t["net_pas_ticks"] for t in trades)
        daily_pnl_mkt.append(day_mkt)
        daily_pnl_pas.append(day_pas)

    if not all_trades:
        return {
            "n_trades": 0, "n_days": len(dates),
            "gross_per_trade": 0, "net_mkt_per_trade": 0, "net_pas_per_trade": 0,
            "wr_mkt": 0, "wr_pas": 0,
            "sharpe_mkt": 0, "sharpe_pas": 0,
            "sortino_mkt": 0, "sortino_pas": 0,
            "pf_mkt": 0, "pf_pas": 0,
            "trades_per_day": 0,
        }

    gross = [t["gross_ticks"] for t in all_trades]
    net_mkt = [t["net_mkt_ticks"] for t in all_trades]
    net_pas = [t["net_pas_ticks"] for t in all_trades]

    n_trades = len(all_trades)

    def _sharpe_sortino(daily_vals):
        if len(daily_vals) < 2:
            return 0.0, 0.0
        dm = np.mean(daily_vals)
        ds = np.std(daily_vals, ddof=1)
        sharpe = dm / ds * np.sqrt(252) if ds > 1e-9 else 0
        neg = [d for d in daily_vals if d < 0]
        dd = np.std(neg, ddof=1) if len(neg) > 1 else ds
        sortino = dm / dd * np.sqrt(252) if dd > 1e-9 else 0
        return float(sharpe), float(sortino)

    def _pf(vals):
        w = sum(v for v in vals if v > 0)
        l = abs(sum(v for v in vals if v < 0))
        return w / l if l > 0 else float("inf")

    sh_mkt, so_mkt = _sharpe_sortino(daily_pnl_mkt)
    sh_pas, so_pas = _sharpe_sortino(daily_pnl_pas)

    return {
        "n_trades": n_trades,
        "n_days": len(dates),
        "trades_per_day": n_trades / max(len(dates), 1),
        "gross_per_trade": float(np.mean(gross)),
        "net_mkt_per_trade": float(np.mean(net_mkt)),
        "net_pas_per_trade": float(np.mean(net_pas)),
        "wr_mkt": sum(1 for v in net_mkt if v > 0) / n_trades,
        "wr_pas": sum(1 for v in net_pas if v > 0) / n_trades,
        "sharpe_mkt": sh_mkt,
        "sharpe_pas": sh_pas,
        "sortino_mkt": so_mkt,
        "sortino_pas": so_pas,
        "pf_mkt": float(_pf(net_mkt)),
        "pf_pas": float(_pf(net_pas)),
        "total_gross": float(sum(gross)),
        "total_mkt": float(sum(net_mkt)),
        "total_pas": float(sum(net_pas)),
    }


def load_day_lean(date_str: str):
    """Load only what's needed: predictions, labels_1s, timestamps (no full events array)."""
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"

    pred_data = np.load(pred_path, allow_pickle=True)
    # Only load labels and timestamps, skip heavy events array
    mbo_data = np.load(mbo_path, mmap_mode="r")

    predictions = pred_data["predictions"]  # (N, 3)
    labels_1s = np.array(mbo_data["labels_1s"])  # copy from mmap
    timestamps = np.array(mbo_data["timestamps"])

    # Trim predictions if needed
    n_preds = predictions.shape[0]
    n_events = len(labels_1s)
    expected_events = (n_preds - 1) * STRIDE + WINDOW_SIZE
    if expected_events > n_events:
        max_preds = (n_events - WINDOW_SIZE) // STRIDE + 1
        predictions = predictions[:max_preds]

    return predictions, labels_1s, timestamps


def run_sweep():
    """Run the full parameter sweep."""
    dates = find_overlap_dates()
    print(f"Found {len(dates)} overlap dates: {dates[0]} to {dates[-1]}")

    # Validate which dates load successfully (one at a time to save memory)
    print("Validating dates...")
    valid_dates = []
    for date_str in dates:
        try:
            pred_path = PRED_DIR / f"{date_str}_predictions.npz"
            mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
            if pred_path.exists() and mbo_path.exists():
                valid_dates.append(date_str)
        except Exception as e:
            print(f"  WARNING: Skipping {date_str}: {e}")

    print(f"Validated {len(valid_dates)} dates")

    # ── Sweep parameters ─────────────────────────────────────────────────
    confidence_pcts = [1, 2, 5, 10]
    cancel_timeouts_ms = [250, 500, 1000]
    mid_move_thresholds = [0.25, 0.5, 1.0]
    max_reprices_list = [0, 1, 2]

    total_combos = len(confidence_pcts) * len(cancel_timeouts_ms) * len(mid_move_thresholds) * len(max_reprices_list)
    print(f"\n{'='*80}")
    print(f"FIFO Cancel/Reprice Sweep: {total_combos} parameter combos x {len(valid_dates)} days")
    print(f"{'='*80}")

    # ── Process one day at a time to save memory ─────────────────────────
    # For each day, run ALL configs and store trade-level results
    # Structure: config_key -> list of (date, day_result)

    all_configs = []
    for conf_pct in confidence_pcts:
        # Baseline config
        all_configs.append(("baseline", conf_pct, 0, 0, 0))
        # Sweep configs
        for cancel_ms in cancel_timeouts_ms:
            for mid_thr in mid_move_thresholds:
                for max_rp in max_reprices_list:
                    all_configs.append(("cancel_rp", conf_pct, cancel_ms, mid_thr, max_rp))

    # Initialize results storage: config_idx -> list of (date, result_dict)
    config_results = {i: [] for i in range(len(all_configs))}

    print(f"\nProcessing {len(valid_dates)} days, {len(all_configs)} configs per day...")

    for day_idx, date_str in enumerate(sorted(valid_dates)):
        try:
            predictions, labels_1s, timestamps = load_day_lean(date_str)
        except Exception as e:
            print(f"  WARNING: Failed to load {date_str}: {e}")
            continue

        if day_idx % 10 == 0:
            print(f"  Day {day_idx+1}/{len(valid_dates)}: {date_str} ({predictions.shape[0]} predictions)")

        # Run all configs on this day
        for cfg_idx, (cfg_type, conf_pct, cancel_ms, mid_thr, max_rp) in enumerate(all_configs):
            if cfg_type == "baseline":
                result = simulate_day_baseline(predictions, labels_1s, timestamps, conf_pct)
            else:
                # simulate_day needs events but we're not loading them for memory
                # Pass None for events - we don't actually use it in the simulation
                result = simulate_day(
                    predictions, labels_1s, timestamps, None,
                    confidence_pct=conf_pct,
                    cancel_timeout_ms=cancel_ms,
                    mid_move_threshold=mid_thr,
                    max_reprices=max_rp,
                )

            if result["trades"]:
                config_results[cfg_idx].append((date_str, result))

        # Explicit cleanup
        del predictions, labels_1s, timestamps

    # ── Compute metrics for all configs ──────────────────────────────────
    print("\n── BASELINES (passive limit, no cancel/reprice) ──")
    print(f"  {'Conf%':>5} {'Trades':>6} {'T/Day':>5} | {'Gross':>7} {'MktExit':>8} {'PasExit':>8} | {'WR_M':>5} {'WR_P':>5} | {'Sh_M':>6} {'Sh_P':>6} {'PF_M':>5} {'PF_P':>5}")
    baselines = {}
    baseline_indices = {}
    for cfg_idx, (cfg_type, conf_pct, _, _, _) in enumerate(all_configs):
        if cfg_type != "baseline":
            continue

        pairs = config_results[cfg_idx]
        dates_list = [d for d, _ in pairs]
        results_list = [r for _, r in pairs]
        metrics = compute_metrics(results_list, dates_list)
        baselines[conf_pct] = metrics
        baseline_indices[conf_pct] = cfg_idx

        pf_m = f"{metrics['pf_mkt']:.2f}" if metrics['pf_mkt'] < 100 else "inf"
        pf_p = f"{metrics['pf_pas']:.2f}" if metrics['pf_pas'] < 100 else "inf"
        print(f"  {conf_pct:5d} {metrics['n_trades']:6d} {metrics['trades_per_day']:5.1f} | "
              f"{metrics['gross_per_trade']:+7.3f} {metrics['net_mkt_per_trade']:+8.3f} {metrics['net_pas_per_trade']:+8.3f} | "
              f"{metrics['wr_mkt']:5.1%} {metrics['wr_pas']:5.1%} | "
              f"{metrics['sharpe_mkt']:6.2f} {metrics['sharpe_pas']:6.2f} {pf_m:>5} {pf_p:>5}")

    # ── Sweep results ────────────────────────────────────────────────────
    print(f"\n── CANCEL/REPRICE SWEEP (MktExit = market order exit, PasExit = passive bid exit) ──")
    print(f"{'Conf%':>5} {'TMs':>4} {'MThr':>4} {'RP':>2} | "
          f"{'Trades':>6} {'T/D':>4} {'Gross':>6} {'MktEx':>6} {'PasEx':>6} {'WR_M':>5} {'WR_P':>5} "
          f"{'Sh_M':>6} {'Sh_P':>6} | {'Δ_M':>6} {'Δ_P':>6}")
    print("-" * 115)

    sweep_results = []

    for cfg_idx, (cfg_type, conf_pct, cancel_ms, mid_thr, max_rp) in enumerate(all_configs):
        if cfg_type == "baseline":
            continue

        pairs = config_results[cfg_idx]
        dates_list = [d for d, _ in pairs]
        results_list = [r for _, r in pairs]
        metrics = compute_metrics(results_list, dates_list)

        # Compare to baseline
        base = baselines.get(conf_pct, {})
        base_mkt = base.get("net_mkt_per_trade", 0)
        base_pas = base.get("net_pas_per_trade", 0)
        imp_mkt = metrics["net_mkt_per_trade"] - base_mkt
        imp_pas = metrics["net_pas_per_trade"] - base_pas

        config_key = f"conf{conf_pct}_cancel{cancel_ms}_mid{mid_thr}_rp{max_rp}"
        entry = {
            "config": config_key,
            "confidence_pct": conf_pct,
            "cancel_timeout_ms": cancel_ms,
            "mid_move_threshold": mid_thr,
            "max_reprices": max_rp,
            "metrics": metrics,
            "improvement_mkt": float(imp_mkt),
            "improvement_pas": float(imp_pas),
        }
        sweep_results.append(entry)

        print(f"{conf_pct:5d} {cancel_ms:4d} {mid_thr:4.1f} {max_rp:2d} | "
              f"{metrics['n_trades']:6d} {metrics['trades_per_day']:4.0f} "
              f"{metrics['gross_per_trade']:+6.2f} {metrics['net_mkt_per_trade']:+6.2f} {metrics['net_pas_per_trade']:+6.2f} "
              f"{metrics['wr_mkt']:5.1%} {metrics['wr_pas']:5.1%} "
              f"{metrics['sharpe_mkt']:6.1f} {metrics['sharpe_pas']:6.1f} | "
              f"{imp_mkt:+6.2f} {imp_pas:+6.2f}")

    # ── Find best configs ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGS by MARKET-EXIT improvement (min 50 trades)")
    print(f"{'='*80}")

    viable = [r for r in sweep_results if r["metrics"]["n_trades"] >= 50]
    viable.sort(key=lambda x: x["improvement_mkt"], reverse=True)

    for i, r in enumerate(viable[:10]):
        m = r["metrics"]
        print(f"  #{i+1}: {r['config']}")
        print(f"      Gross: {m['gross_per_trade']:+.3f} | MktExit: {m['net_mkt_per_trade']:+.3f} | PasExit: {m['net_pas_per_trade']:+.3f}")
        print(f"      WR(mkt): {m['wr_mkt']:.1%} | WR(pas): {m['wr_pas']:.1%} | Trades: {m['n_trades']} ({m['trades_per_day']:.0f}/day)")
        print(f"      Sharpe(mkt): {m['sharpe_mkt']:.2f} | Sharpe(pas): {m['sharpe_pas']:.2f}")
        print(f"      Δ vs baseline: mkt={r['improvement_mkt']:+.3f}, pas={r['improvement_pas']:+.3f}")

    # ── Summary comparison table ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("BASELINE vs BEST CANCEL/REPRICE (per confidence level)")
    print(f"{'='*80}")

    for conf_pct in confidence_pcts:
        base = baselines.get(conf_pct, {})
        conf_viable = [r for r in viable if r["confidence_pct"] == conf_pct]

        print(f"\n  Top {conf_pct}% confidence:")
        print(f"    Baseline:       Gross {base.get('gross_per_trade',0):+.3f} | "
              f"MktExit {base.get('net_mkt_per_trade',0):+.3f} | PasExit {base.get('net_pas_per_trade',0):+.3f} | "
              f"WR(m/p) {base.get('wr_mkt',0):.1%}/{base.get('wr_pas',0):.1%} | "
              f"Trades {base.get('n_trades', 0)}")

        if conf_viable:
            best = conf_viable[0]
            m = best["metrics"]
            print(f"    Best cancel/RP: Gross {m['gross_per_trade']:+.3f} | "
                  f"MktExit {m['net_mkt_per_trade']:+.3f} | PasExit {m['net_pas_per_trade']:+.3f} | "
                  f"WR(m/p) {m['wr_mkt']:.1%}/{m['wr_pas']:.1%} | "
                  f"Trades {m['n_trades']}")
            print(f"    Config: cancel={best['cancel_timeout_ms']}ms, "
                  f"mid_thr={best['mid_move_threshold']}, "
                  f"max_rp={best['max_reprices']}")
            print(f"    Δ mkt: {best['improvement_mkt']:+.3f} | Δ pas: {best['improvement_pas']:+.3f}")
        else:
            print(f"    No viable cancel/reprice configs found")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "run_date": "2026-05-23",
        "n_dates": len(valid_dates),
        "date_range": f"{valid_dates[0]} to {valid_dates[-1]}",
        "cost_model": {
            "commission_rt_ticks": COMMISSION_TICKS_RT,
            "market_exit_cost": COST_MARKET_EXIT,
            "passive_exit_cost": COST_PASSIVE_EXIT,
            "note": "Passive entry at ask + market exit at ask = cost is commission only. Passive exit at bid earns spread.",
        },
        "baselines": {
            f"top_{pct}pct": metrics
            for pct, metrics in baselines.items()
        },
        "sweep_results": sweep_results,
        "best_configs_mkt_exit": [
            {
                "rank": i + 1,
                "config": r["config"],
                "confidence_pct": r["confidence_pct"],
                "cancel_timeout_ms": r["cancel_timeout_ms"],
                "mid_move_threshold": r["mid_move_threshold"],
                "max_reprices": r["max_reprices"],
                "gross_per_trade": r["metrics"]["gross_per_trade"],
                "net_mkt_per_trade": r["metrics"]["net_mkt_per_trade"],
                "net_pas_per_trade": r["metrics"]["net_pas_per_trade"],
                "wr_mkt": r["metrics"]["wr_mkt"],
                "sharpe_mkt": r["metrics"]["sharpe_mkt"],
                "improvement_mkt": r["improvement_mkt"],
            }
            for i, r in enumerate(viable[:10])
        ],
    }

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return output


if __name__ == "__main__":
    t0 = time.time()
    run_sweep()
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
