"""
mbo_creative_signals.py
========================
Explore 3 creative MBO-derived signals that we have NOT tried before.
Uses raw mbo_events (6 cols) + timestamps from processed/mbo_events/ directory.

Signals:
  1. Queue Depletion Velocity — rate of consumption at best bid/ask
  2. Cancel-to-Trade Ratio Spikes — spoofing/testing detection
  3. Trade-Sign Persistence (run length) — institutional accumulation detection

For each signal, compute IC against future price at 1s, 5s, 10s, 30s, 60s, 300s horizons.
Uses v4 preprocessed data (has labels at 1s, 5s, 10s, 30s).

Author: Claude Opus 4.6 (quant research)
Date: 2026-08-27
"""

import os
import sys
import time
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
V4_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v4")
RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/strategies")

# Event type constants (from precompute_features_smart_v4.py)
_EVENT_TYPE_ADD    = 0
_EVENT_TYPE_CANCEL = 1
_EVENT_TYPE_MODIFY = 2
_EVENT_TYPE_TRADE  = 3
_EVENT_TYPE_FILL   = 4
_SIDE_BID = 0
_SIDE_ASK = 1

# Horizons available in v4 labels
LABEL_HORIZONS = ["1s", "5s", "10s", "30s"]

# Sample parameters — use N days for speed, full set for final
MAX_DAYS = 30  # use up to 30 days for robust estimates


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}  {msg}", flush=True)


# ============================================================
# Helpers
# ============================================================
def causal_rolling_sum(signal: np.ndarray, W: int) -> np.ndarray:
    N = len(signal)
    cs = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64))))
    idx_end = np.arange(1, N + 1, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    return (cs[idx_end] - cs[idx_start]).astype(np.float32)


def causal_rolling_mean(signal: np.ndarray, W: int) -> np.ndarray:
    N = len(signal)
    cs = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64))))
    idx_end = np.arange(1, N + 1, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    counts = (idx_end - idx_start).astype(np.float64)
    counts[counts == 0] = 1.0
    return ((cs[idx_end] - cs[idx_start]) / counts).astype(np.float32)


def ewma_fast(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized EWMA approximation using cumsum trick for speed."""
    N = len(signal)
    out = np.zeros(N, dtype=np.float64)
    out[0] = signal[0]
    for i in range(1, min(N, 50000)):  # do loop for first chunk
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
    # Continue from last computed
    for i in range(50000, N):
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
    return out.astype(np.float32)


def rank_ic(signal: np.ndarray, label: np.ndarray, max_sample: int = 500_000) -> Tuple[float, float, int]:
    """Compute rank IC (Spearman correlation) between signal and label.
    Subsamples to max_sample for speed. Returns (ic, t_stat, n_obs)."""
    valid = np.isfinite(signal) & np.isfinite(label) & (signal != 0)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return np.nan, np.nan, n_valid
    s = signal[valid]
    l = label[valid]
    # Subsample for speed if too many observations
    if n_valid > max_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_valid, max_sample, replace=False)
        s = s[idx]
        l = l[idx]
        n_used = max_sample
    else:
        n_used = n_valid
    ic, pval = stats.spearmanr(s, l)
    t = ic * np.sqrt((n_used - 2) / (1 - ic**2 + 1e-12))
    return float(ic), float(t), n_valid


def subsample_for_speed(events, timestamps, labels_dict, max_events=2_000_000):
    """Subsample events to keep computation tractable.
    Take every Nth event to maintain time coverage."""
    N = events.shape[0]
    if N <= max_events:
        return events, timestamps, labels_dict
    step = N // max_events
    idx = np.arange(0, N, step)[:max_events]
    sub_labels = {}
    for k, v in labels_dict.items():
        sub_labels[k] = v[idx]
    return events[idx], timestamps[idx], sub_labels


# ============================================================
# Signal 1: Queue Depletion Velocity
# ============================================================
def compute_queue_depletion_velocity(events: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Queue Depletion Velocity: how fast resting orders are being consumed at best bid/ask.

    Different from OFI:
    - OFI = net signed volume flow (adds vs cancels vs fills)
    - QDV = RATE of depletion (fills + cancels per unit time) on each side

    Key insight: if bid-side orders are being consumed rapidly (lots of fills + cancels at bid)
    while ask-side is stable, price is about to drop. The RATE matters more than the level.

    We compute:
    - bid_depletion_rate: rolling rate of (fills + cancels) on bid side per time unit
    - ask_depletion_rate: rolling rate of (fills + cancels) on ask side per time unit
    - qdv_asymmetry: (ask_depletion - bid_depletion) / total, normalized
    - qdv_acceleration: rate of change of qdv_asymmetry
    """
    N = events.shape[0]
    event_type = events[:, 1]
    side = events[:, 2]
    qty_log = events[:, 4].astype(np.float64)
    time_delta_log = events[:, 0].astype(np.float64)

    # Events that deplete the queue: fills, trades, and cancels
    is_depletion = ((event_type == _EVENT_TYPE_FILL) |
                    (event_type == _EVENT_TYPE_TRADE) |
                    (event_type == _EVENT_TYPE_CANCEL)).astype(np.float64)

    is_bid = (side == _SIDE_BID).astype(np.float64)
    is_ask = (side == _SIDE_ASK).astype(np.float64)

    # Volume-weighted depletion on each side
    bid_depletion = is_depletion * is_bid * qty_log
    ask_depletion = is_depletion * is_ask * qty_log

    signals = {}

    for W in [50, 200, 500]:
        bid_dep_sum = causal_rolling_sum(bid_depletion, W)
        ask_dep_sum = causal_rolling_sum(ask_depletion, W)
        total = bid_dep_sum + ask_dep_sum
        total[total < 0.01] = 0.01

        # Asymmetry: positive = more ask depletion = bullish (asks getting hit)
        asym = (ask_dep_sum - bid_dep_sum) / total
        signals[f"qdv_asym_{W}"] = asym

        # Acceleration (rate of change of asymmetry)
        diff = np.zeros(N, dtype=np.float32)
        lag = min(W // 5, 50)
        diff[lag:] = asym[lag:] - asym[:-lag]
        signals[f"qdv_accel_{W}"] = diff

    # Also compute: RATE of depletion (time-normalized)
    # Convert time_delta_log to approximate seconds
    time_approx = np.exp(time_delta_log)  # undo log
    elapsed = causal_rolling_sum(time_approx, 200)
    elapsed[elapsed < 1e-6] = 1e-6

    bid_rate = causal_rolling_sum(bid_depletion, 200) / elapsed
    ask_rate = causal_rolling_sum(ask_depletion, 200) / elapsed
    total_rate = bid_rate + ask_rate
    total_rate[total_rate < 1e-6] = 1e-6

    signals["qdv_rate_asym"] = (ask_rate - bid_rate) / total_rate

    return signals


# ============================================================
# Signal 2: Cancel-to-Trade Ratio Spikes
# ============================================================
def compute_cancel_trade_ratio(events: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Cancel-to-Trade Ratio (CTR) spikes.

    High cancellation rate relative to fills = someone testing the market / spoofing.
    The key insight: AFTER a CTR spike, look at which side's orders survived.
    If bid orders survived while ask orders got cancelled → bearish pressure was fake → bullish.

    We compute:
    - ctr_spike: deviation of cancel-to-trade ratio from its rolling mean
    - ctr_directional: after spike, which side had more surviving orders (net direction)
    - ctr_spike_x_direction: interaction of spike intensity with direction
    """
    N = events.shape[0]
    event_type = events[:, 1]
    side = events[:, 2]
    qty_log = events[:, 4].astype(np.float64)

    is_cancel = (event_type == _EVENT_TYPE_CANCEL).astype(np.float64)
    is_trade = ((event_type == _EVENT_TYPE_TRADE) |
                (event_type == _EVENT_TYPE_FILL)).astype(np.float64)
    is_add = (event_type == _EVENT_TYPE_ADD).astype(np.float64)

    is_bid = (side == _SIDE_BID).astype(np.float64)
    is_ask = (side == _SIDE_ASK).astype(np.float64)

    signals = {}

    for W in [50, 200, 500]:
        # Cancel-to-trade ratio
        cancel_count = causal_rolling_sum(is_cancel, W)
        trade_count = causal_rolling_sum(is_trade, W)
        trade_count_safe = np.maximum(trade_count, 1.0)
        ctr = cancel_count / trade_count_safe

        # Z-score of CTR vs longer lookback
        ctr_mean = causal_rolling_mean(ctr, W * 10)
        ctr_diff = ctr - ctr_mean

        signals[f"ctr_spike_{W}"] = ctr_diff

        # Directional: which side is cancelling more?
        bid_cancel = causal_rolling_sum(is_cancel * is_bid * qty_log, W)
        ask_cancel = causal_rolling_sum(is_cancel * is_ask * qty_log, W)
        total_cancel = bid_cancel + ask_cancel
        total_cancel[total_cancel < 0.01] = 0.01

        # If ask cancellations dominate → sellers pulling out → bullish
        cancel_dir = (ask_cancel - bid_cancel) / total_cancel
        signals[f"ctr_direction_{W}"] = cancel_dir

        # Interaction: spike intensity x direction
        signals[f"ctr_spike_x_dir_{W}"] = ctr_diff * cancel_dir

    # Post-cancel residual: after cancels, are adds replenishing one side more?
    # Add-after-cancel asymmetry
    for W in [100, 500]:
        bid_add = causal_rolling_sum(is_add * is_bid * qty_log, W)
        ask_add = causal_rolling_sum(is_add * is_ask * qty_log, W)
        bid_cancel_w = causal_rolling_sum(is_cancel * is_bid * qty_log, W)
        ask_cancel_w = causal_rolling_sum(is_cancel * is_ask * qty_log, W)

        # Net queue change per side (add - cancel)
        bid_net = bid_add - bid_cancel_w
        ask_net = ask_add - ask_cancel_w
        total_abs = np.abs(bid_net) + np.abs(ask_net)
        total_abs[total_abs < 0.01] = 0.01

        # Positive = ask side strengthening relative to bid = bearish
        signals[f"queue_net_asym_{W}"] = (ask_net - bid_net) / total_abs

    return signals


# ============================================================
# Signal 3: Trade-Sign Persistence (Run Analysis)
# ============================================================
def compute_trade_sign_persistence(events: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Trade-Sign Persistence: detect runs of consecutive same-direction trades.

    Key insight: Institutional accumulation creates runs of 10-50+ consecutive buys (or sells).
    Retail flow is random. Long runs = institutional activity = directional signal.

    We compute:
    - run_length: current length of same-direction trade run
    - run_intensity: cumulative volume of current run
    - run_signed: run_length * direction (+1 buy, -1 sell)
    - run_ratio: fraction of recent trades in same direction as current run
    - run_break_signal: what happens at run termination (reversal prediction)
    """
    N = events.shape[0]
    event_type = events[:, 1]
    side = events[:, 2]
    qty_log = events[:, 4].astype(np.float64)
    time_delta_log = events[:, 0].astype(np.float64)

    is_trade = ((event_type == _EVENT_TYPE_TRADE) |
                (event_type == _EVENT_TYPE_FILL)).astype(np.float64)
    is_bid = (side == _SIDE_BID).astype(np.float64)
    is_ask = (side == _SIDE_ASK).astype(np.float64)

    # Trade direction: +1 for buy (aggressor hits ask), -1 for sell (aggressor hits bid)
    # In MBO data: a trade on the ASK side means someone bought (lifted the offer)
    # a trade on the BID side means someone sold (hit the bid)
    trade_sign = is_trade * (is_ask - is_bid)  # +1 buy, -1 sell

    # Compute run length and intensity
    run_length = np.zeros(N, dtype=np.float32)
    run_intensity = np.zeros(N, dtype=np.float32)
    run_direction = np.zeros(N, dtype=np.float32)  # direction of current run

    current_dir = 0.0
    current_len = 0.0
    current_vol = 0.0

    for i in range(N):
        if trade_sign[i] != 0:
            if trade_sign[i] == current_dir:
                current_len += 1
                current_vol += qty_log[i]
            else:
                current_dir = trade_sign[i]
                current_len = 1
                current_vol = qty_log[i]

        run_length[i] = current_len
        run_intensity[i] = current_vol
        run_direction[i] = current_dir

    signals = {}

    # Raw run features
    signals["run_length_signed"] = run_length * run_direction
    signals["run_intensity_signed"] = run_intensity * run_direction

    # Log-scaled run length to prevent outlier dominance
    signals["run_length_log_signed"] = np.log1p(run_length) * run_direction

    # Rolling fraction of trades in one direction
    for W in [50, 200, 500]:
        buy_count = causal_rolling_sum((trade_sign > 0).astype(np.float64), W)
        sell_count = causal_rolling_sum((trade_sign < 0).astype(np.float64), W)
        total = buy_count + sell_count
        total[total < 1] = 1
        # Buy fraction - 0.5, so centered around 0
        signals[f"trade_dir_frac_{W}"] = (buy_count / total - 0.5).astype(np.float32) * 2

    # Run break prediction: when a long run ends, reversal or continuation?
    # Detect run breaks: direction flips on trade events
    run_break = np.zeros(N, dtype=np.float64)
    prev_dir = 0.0
    for i in range(N):
        if trade_sign[i] != 0:
            if prev_dir != 0 and trade_sign[i] != prev_dir:
                run_break[i] = prev_dir  # direction of the BROKEN run
            prev_dir = trade_sign[i]

    # Rolling run break signal: recent run breaks tend to cluster
    for W in [100, 500]:
        signals[f"run_break_sum_{W}"] = causal_rolling_sum(run_break, W)

    return signals


# ============================================================
# Main: Load data, compute signals, measure IC
# ============================================================
def load_v4_files(max_days: int = MAX_DAYS) -> List[Path]:
    """Get list of v4 data files, sorted by date."""
    files = sorted(V4_DIR.glob("*_mbo_events.npz"))
    if len(files) > max_days:
        # Take evenly spaced sample for better date coverage
        step = len(files) / max_days
        files = [files[int(i * step)] for i in range(max_days)]
    return files


def analyze_signals():
    """Main analysis: compute all creative signals and measure IC."""
    files = load_v4_files()
    log(f"Loaded {len(files)} day files from {V4_DIR}")

    if len(files) == 0:
        log("ERROR: No data files found!")
        return

    # Results storage
    all_results = {}

    # Process each day
    signal_accum = {}  # signal_name -> list of (signal_chunk, label_chunks)

    for fi, fpath in enumerate(files):
        date_str = fpath.stem.split("_")[0]
        log(f"[{fi+1}/{len(files)}] Processing {date_str}...")

        try:
            d = np.load(fpath, allow_pickle=True)
            events = d["events"]       # (N, 29) but we need raw cols 0-5
            event_type_raw = d["event_type_raw"]  # (N,) int8
            timestamps = d["timestamps"]

            labels = {}
            for h in LABEL_HORIZONS:
                key = f"labels_{h}"
                if key in d:
                    labels[h] = d[key]

            N = events.shape[0]
            log(f"  {N:,} events, labels: {list(labels.keys())}")

            # Reconstruct raw-style events from v4 features
            # v4 features 0-5 are the original 6 raw features
            raw_events = events[:, :6].copy()
            # But event_type_raw gives us the actual type
            raw_events[:, 1] = event_type_raw.astype(np.float32)

            # Subsample for speed on large days
            raw_sub, ts_sub, labels_sub = subsample_for_speed(
                raw_events, timestamps, labels, max_events=1_500_000
            )

            # Compute all signal families
            log(f"  Computing Queue Depletion Velocity...")
            qdv_signals = compute_queue_depletion_velocity(raw_sub, ts_sub)

            log(f"  Computing Cancel-to-Trade Ratio...")
            ctr_signals = compute_cancel_trade_ratio(raw_sub, ts_sub)

            log(f"  Computing Trade-Sign Persistence...")
            tsp_signals = compute_trade_sign_persistence(raw_sub, ts_sub)

            # Merge all signals
            all_signals = {}
            all_signals.update(qdv_signals)
            all_signals.update(ctr_signals)
            all_signals.update(tsp_signals)

            # Accumulate for cross-day IC
            for sig_name, sig_vals in all_signals.items():
                if sig_name not in signal_accum:
                    signal_accum[sig_name] = {h: ([], []) for h in LABEL_HORIZONS}
                for h in LABEL_HORIZONS:
                    if h in labels_sub:
                        signal_accum[sig_name][h][0].append(sig_vals)
                        signal_accum[sig_name][h][1].append(labels_sub[h])

        except Exception as e:
            log(f"  ERROR: {e}")
            continue

    # ============================================================
    # Compute pooled IC across all days
    # ============================================================
    log("\n" + "=" * 80)
    log("RESULTS: Rank IC (Spearman) by signal and horizon")
    log("=" * 80)

    results_table = []

    for sig_name in sorted(signal_accum.keys()):
        row = {"signal": sig_name}
        for h in LABEL_HORIZONS:
            sigs_list, labs_list = signal_accum[sig_name][h]
            if len(sigs_list) == 0:
                row[f"IC_{h}"] = np.nan
                row[f"t_{h}"] = np.nan
                row[f"n_{h}"] = 0
                continue

            sig_all = np.concatenate(sigs_list)
            lab_all = np.concatenate(labs_list)

            ic, t, n = rank_ic(sig_all, lab_all)
            row[f"IC_{h}"] = ic
            row[f"t_{h}"] = t
            row[f"n_{h}"] = n

        results_table.append(row)

    # Print results sorted by absolute IC at 1s
    results_table.sort(key=lambda r: abs(r.get("IC_1s", 0) or 0), reverse=True)

    # Print header
    header = f"{'Signal':<30s}"
    for h in LABEL_HORIZONS:
        header += f"  IC_{h:>3s}   t_{h:>3s}"
    log(header)
    log("-" * 120)

    for row in results_table:
        line = f"{row['signal']:<30s}"
        for h in LABEL_HORIZONS:
            ic = row.get(f"IC_{h}", np.nan)
            t = row.get(f"t_{h}", np.nan)
            if np.isfinite(ic):
                line += f"  {ic:+.4f}  {t:+7.1f}"
            else:
                line += f"     NaN      NaN"
        log(line)

    # ============================================================
    # Summary: which signals are tradeable?
    # ============================================================
    log("\n" + "=" * 80)
    log("SIGNAL QUALITY ASSESSMENT")
    log("=" * 80)

    # Cost thresholds:
    # Passive limit: 0.376 ticks RT commission -> need ~0.5 ticks edge
    # Market order: 1.376 ticks RT -> need ~1.5 ticks edge
    # IC > 0.02 at 1s is interesting, IC > 0.05 is strong, IC > 0.10 is exceptional

    strong_signals = []
    interesting_signals = []

    for row in results_table:
        name = row["signal"]
        best_ic = 0
        best_h = ""
        for h in LABEL_HORIZONS:
            ic = abs(row.get(f"IC_{h}", 0) or 0)
            t = abs(row.get(f"t_{h}", 0) or 0)
            if ic > best_ic and t > 3.0:  # require t > 3 for significance
                best_ic = ic
                best_h = h

        if best_ic >= 0.05:
            strong_signals.append((name, best_ic, best_h))
        elif best_ic >= 0.02:
            interesting_signals.append((name, best_ic, best_h))

    log(f"\nSTRONG signals (|IC| >= 0.05, |t| > 3):")
    for name, ic, h in sorted(strong_signals, key=lambda x: -x[1]):
        log(f"  {name:<30s}  IC={ic:.4f} at {h}")

    log(f"\nINTERESTING signals (0.02 <= |IC| < 0.05, |t| > 3):")
    for name, ic, h in sorted(interesting_signals, key=lambda x: -x[1]):
        log(f"  {name:<30s}  IC={ic:.4f} at {h}")

    weak = len(results_table) - len(strong_signals) - len(interesting_signals)
    log(f"\nWeak/insignificant: {weak} signals")

    # ============================================================
    # Decay analysis for top signals
    # ============================================================
    log("\n" + "=" * 80)
    log("DECAY ANALYSIS: How does IC change across horizons for top signals?")
    log("=" * 80)

    top_n = min(10, len(results_table))
    for row in results_table[:top_n]:
        name = row["signal"]
        ics = []
        for h in LABEL_HORIZONS:
            ic = row.get(f"IC_{h}", 0) or 0
            ics.append(f"{h}:{ic:+.4f}")
        log(f"  {name:<30s}  {' | '.join(ics)}")

    # ============================================================
    # Per-day IC stability check for top signals
    # ============================================================
    log("\n" + "=" * 80)
    log("STABILITY: Per-day IC distribution for top 5 signals")
    log("=" * 80)

    # Recompute per-day IC for top signals
    top_signal_names = [r["signal"] for r in results_table[:5]]

    for sig_name in top_signal_names:
        if sig_name not in signal_accum:
            continue

        # Per-day IC at 1s horizon
        h = "1s"
        sigs_list, labs_list = signal_accum[sig_name][h]
        day_ics = []
        for s, l in zip(sigs_list, labs_list):
            ic, t, n = rank_ic(s, l)
            if np.isfinite(ic):
                day_ics.append(ic)

        if len(day_ics) >= 5:
            day_ics = np.array(day_ics)
            pct_pos = (day_ics > 0).mean() * 100
            log(f"  {sig_name:<30s}  mean={np.mean(day_ics):+.4f}  "
                f"std={np.std(day_ics):.4f}  "
                f"median={np.median(day_ics):+.4f}  "
                f"pct_positive={pct_pos:.0f}%  "
                f"n_days={len(day_ics)}")

    # Save results to JSON
    output_path = OUTPUT_DIR / "mbo_creative_signals_results.json"
    json_results = []
    for row in results_table:
        json_row = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, float)):
                json_row[k] = round(float(v), 6) if np.isfinite(v) else None
            elif isinstance(v, (np.integer, int)):
                json_row[k] = int(v)
            else:
                json_row[k] = v
        json_results.append(json_row)

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {output_path}")

    return results_table


if __name__ == "__main__":
    log("=" * 80)
    log("MBO Creative Signals Analysis")
    log("Signals: Queue Depletion Velocity, Cancel-to-Trade Ratio, Trade-Sign Persistence")
    log("=" * 80)
    analyze_signals()
