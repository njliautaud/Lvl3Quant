"""
precompute_features_smart_v2.py
===============================
Smart per-feature preprocessing V2 — adds 7 high-IC features from empirical testing
and academic literature to the original 15, for 22 total features.

NEW FEATURES (15-21):
  15: queue_replenishment     — add_rate / cancel_rate (100 events). IC=0.074
  16: mom_divergence          — short vs long momentum divergence. IC=0.066
  17: ofi_x_spread            — OFI × spread (non-linear interaction). IC=0.053
  18: vol_weighted_pmom       — volume-weighted price momentum. IC=0.053
  19: buy_sell_intensity_ratio — EWMA event rate ratio by side. IC≈0.03 (literature)
  20: realized_volatility     — rolling sqrt(mean(price_change²)). IC≈0.02 (literature)
  21: sweep_intensity         — 3+ same-side trades in rapid succession. IC≈0.04 (literature)

ORIGINAL 15 FEATURES (same treatment as v1):
  0-14: Same as precompute_features_smart.py (smart per-feature normalization)

Output: data/processed/mbo_events_smart_v2/ — (N, 22) float32
Training: MAMBA_FEATURE_SET=smart_v2, normalize_features=False

Usage:
    python precompute_features_smart_v2.py

    # Custom dirs:
    SMART_INPUT_DIR=... SMART_OUTPUT_DIR=... python precompute_features_smart_v2.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ============================================================
# Config
# ============================================================
INPUT_DIR = Path(os.environ.get(
    "SMART_INPUT_DIR",
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events"
))
OUTPUT_DIR = Path(os.environ.get(
    "SMART_OUTPUT_DIR",
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2"
))

# Rolling z-score lookback windows (in events)
ROLLING_ZSCORE_LOOKBACK_LONG = 10000
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000

# EWMA decay for intensity features
EWMA_ALPHA = 0.01  # ~100-event half-life

# Sweep detection parameters
SWEEP_MIN_TRADES = 3        # minimum consecutive same-side trades
SWEEP_MAX_GAP_LOG = 0.5     # max time_delta_log between events in a sweep (~1.6x gap)

# Encoder constants
_EVENT_TYPE_ADD    = 0
_EVENT_TYPE_CANCEL = 1
_EVENT_TYPE_MODIFY = 2
_EVENT_TYPE_TRADE  = 3
_EVENT_TYPE_FILL   = 4
_SIDE_BID = 0
_SIDE_ASK = 1
_N_EVENT_TYPES = 5

N_RAW = 6
N_DERIVED_V1 = 9    # Original 9 derived features
N_NEW = 7            # New features in v2
N_FEATURES = N_RAW + N_DERIVED_V1 + N_NEW  # 22


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ============================================================
# Causal rolling helpers
# ============================================================
def causal_rolling_sum(signal: np.ndarray, W: int) -> np.ndarray:
    N = len(signal)
    cs = np.concatenate(([0.0], np.cumsum(signal)))
    idx_end = np.arange(N, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    return (cs[idx_end] - cs[idx_start]).astype(np.float32)


def causal_rolling_mean(signal: np.ndarray, W: int) -> np.ndarray:
    N = len(signal)
    cs = np.concatenate(([0.0], np.cumsum(signal)))
    idx_end = np.arange(N, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    counts = (idx_end - idx_start).astype(np.float32)
    counts[counts == 0] = 1.0
    return ((cs[idx_end] - cs[idx_start]) / counts).astype(np.float32)


def causal_rolling_zscore(signal: np.ndarray, W: int) -> np.ndarray:
    """Causal rolling z-score clipped to [-5, 5]."""
    N = len(signal)
    sig64 = signal.astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(sig64)))
    cs_sq = np.concatenate(([0.0], np.cumsum(sig64 ** 2)))
    idx_end = np.arange(N, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    counts = (idx_end - idx_start).astype(np.float64)
    counts[counts == 0] = 1.0
    sums = cs[idx_end] - cs[idx_start]
    sums_sq = cs_sq[idx_end] - cs_sq[idx_start]
    means = sums / counts
    variances = np.maximum((sums_sq / counts) - (means ** 2), 0.0)
    stds = np.sqrt(variances)
    stds[stds < 1e-8] = 1e-8
    z = (sig64 - means) / stds
    return np.clip(z, -5.0, 5.0).astype(np.float32)


def ewma(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average. O(N), causal."""
    N = len(signal)
    out = np.zeros(N, dtype=np.float64)
    out[0] = signal[0]
    for i in range(1, N):
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
    return out.astype(np.float32)


# ============================================================
# Compute original 9 derived features (same as v1)
# ============================================================
def compute_derived_features_v1(ev: np.ndarray) -> np.ndarray:
    """Returns (N, 9) array of v1 derived features."""
    N = ev.shape[0]
    derived = np.zeros((N, N_DERIVED_V1), dtype=np.float32)

    time_delta_log  = ev[:, 0].astype(np.float64)
    event_type_id   = ev[:, 1]
    side_id         = ev[:, 2]
    price_rel_ticks = ev[:, 3].astype(np.float64)
    qty_log         = ev[:, 4].astype(np.float64)
    spread_ticks    = ev[:, 5].astype(np.float64)

    is_cancel = (event_type_id == _EVENT_TYPE_CANCEL).astype(np.float64)
    is_ask    = (side_id == _SIDE_ASK).astype(np.float64)
    is_bid    = (side_id == _SIDE_BID).astype(np.float64)

    # 0: cancel_side_asym_50
    cancel_signal = is_cancel * (is_ask - is_bid)
    derived[:, 0] = causal_rolling_sum(cancel_signal, 50)

    # 1: rolling_ofi_500
    sign_side = is_ask - is_bid
    ofi_signal = qty_log * sign_side
    derived[:, 1] = causal_rolling_sum(ofi_signal, 500)

    # 2: event_density_20
    derived[:, 2] = causal_rolling_mean(time_delta_log, 20)

    # 3: price_mom_10
    derived[:, 3] = causal_rolling_sum(price_rel_ticks, 10)

    # 4: qty_price_mom_50
    qty_price = qty_log * price_rel_ticks
    derived[:, 4] = causal_rolling_sum(qty_price, 50)

    # 5: price_sign_momentum_200
    price_sign = np.sign(price_rel_ticks)
    derived[:, 5] = causal_rolling_sum(price_sign, 200)

    # 6: event_type_entropy_200
    K = _N_EVENT_TYPES
    W_ENT = 200
    type_cs = np.zeros((K, N + 1), dtype=np.float64)
    for k in range(K):
        indicator = (event_type_id == k).astype(np.float64)
        type_cs[k] = np.concatenate(([0.0], np.cumsum(indicator)))
    idx_end_ent   = np.arange(N, dtype=np.int64)
    idx_start_ent = np.maximum(0, idx_end_ent - W_ENT)
    window_counts = (idx_end_ent - idx_start_ent).astype(np.float64)
    window_counts[window_counts == 0] = 1.0
    entropy = np.zeros(N, dtype=np.float32)
    for k in range(K):
        counts_k = type_cs[k][idx_end_ent] - type_cs[k][idx_start_ent]
        p_k = counts_k / window_counts
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(p_k > 0.0, np.log(p_k), 0.0)
        entropy -= (p_k * log_p).astype(np.float32)
    derived[:, 6] = entropy

    # 7: fill_add_restoration_100
    is_fill_trade = ((event_type_id == _EVENT_TYPE_FILL) |
                     (event_type_id == _EVENT_TYPE_TRADE)).astype(np.float64)
    is_add = (event_type_id == _EVENT_TYPE_ADD).astype(np.float64)
    prev_is_ft = np.empty(N, dtype=np.float64)
    prev_is_ft[0] = 0.0
    prev_is_ft[1:] = is_fill_trade[:-1]
    prev_side = np.empty(N, dtype=np.float64)
    prev_side[0] = -1.0
    prev_side[1:] = side_id[:-1].astype(np.float64)
    same_side = (side_id.astype(np.float64) == prev_side).astype(np.float64)
    fill_add_signal = prev_is_ft * is_add * same_side
    W_FAR = 100
    fa_cs   = np.concatenate(([0.0], np.cumsum(fill_add_signal)))
    ft_cs   = np.concatenate(([0.0], np.cumsum(is_fill_trade)))
    idx_end_far   = np.arange(N, dtype=np.int64)
    idx_start_far = np.maximum(0, idx_end_far - W_FAR)
    pair_counts  = fa_cs[idx_end_far] - fa_cs[idx_start_far]
    fill_counts  = np.maximum(ft_cs[idx_end_far] - ft_cs[idx_start_far], 1.0)
    derived[:, 7] = (pair_counts / fill_counts).astype(np.float32)

    # 8: spread_velocity_50
    spread_diff = np.empty(N, dtype=np.float64)
    spread_diff[0] = 0.0
    spread_diff[1:] = spread_ticks[1:] - spread_ticks[:-1]
    derived[:, 8] = causal_rolling_mean(spread_diff, 50)

    return derived


# ============================================================
# Compute NEW v2 features (7 additional)
# ============================================================
def compute_new_features(ev: np.ndarray) -> np.ndarray:
    """
    Compute 7 new high-IC features. Returns (N, 7).

    15: queue_replenishment     — IC=0.074
    16: mom_divergence          — IC=0.066
    17: ofi_x_spread            — IC=0.053
    18: vol_weighted_pmom       — IC=0.053
    19: buy_sell_intensity_ratio — IC≈0.03
    20: realized_volatility     — IC≈0.02
    21: sweep_intensity         — IC≈0.04
    """
    N = ev.shape[0]
    new = np.zeros((N, N_NEW), dtype=np.float32)

    time_delta_log  = ev[:, 0].astype(np.float64)
    event_type_id   = ev[:, 1]
    side_id         = ev[:, 2]
    price_rel_ticks = ev[:, 3].astype(np.float64)
    qty_log         = ev[:, 4].astype(np.float64)
    spread_ticks    = ev[:, 5].astype(np.float64)

    is_cancel = (event_type_id == _EVENT_TYPE_CANCEL).astype(np.float64)
    is_add    = (event_type_id == _EVENT_TYPE_ADD).astype(np.float64)
    is_trade  = ((event_type_id == _EVENT_TYPE_TRADE) |
                 (event_type_id == _EVENT_TYPE_FILL)).astype(np.float64)
    is_ask    = (side_id == _SIDE_ASK).astype(np.float64)
    is_bid    = (side_id == _SIDE_BID).astype(np.float64)
    sign_side = is_ask - is_bid

    # ------------------------------------------------------------------
    # Feature 15: queue_replenishment (IC=0.074)
    # Ratio of add rate to cancel rate over 100 events.
    # High = market makers actively refilling book. Low = liquidity draining.
    # ------------------------------------------------------------------
    add_rate = causal_rolling_sum(is_add * qty_log, 100)
    cancel_rate = causal_rolling_sum(is_cancel * qty_log, 100)
    new[:, 0] = (add_rate / np.maximum(cancel_rate, 0.1)).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 16: mom_divergence (IC=0.066)
    # Short-term vs long-term price sign momentum divergence.
    # Positive = short-term accelerating vs long-term. Regime change signal.
    # ------------------------------------------------------------------
    price_sign = np.sign(price_rel_ticks)
    pmom_short = causal_rolling_sum(price_sign, 20)
    pmom_long  = causal_rolling_sum(price_sign, 200)
    # Normalize both to per-event rate, then take difference
    new[:, 1] = (pmom_short / 20.0 - pmom_long / 200.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 17: ofi_x_spread (IC=0.053)
    # Non-linear interaction: OFI × spread. Order flow is more informative
    # when spread is wide (illiquid). Classic toxicity-adjusted signal.
    # ------------------------------------------------------------------
    ofi_100 = causal_rolling_sum(qty_log * sign_side, 100)
    new[:, 2] = (ofi_100 * spread_ticks).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 18: vol_weighted_pmom (IC=0.053)
    # Volume-weighted price momentum. Large orders confirming direction.
    # ------------------------------------------------------------------
    new[:, 3] = causal_rolling_sum(qty_log * price_rel_ticks, 50)

    # ------------------------------------------------------------------
    # Feature 19: buy_sell_intensity_ratio (IC≈0.03)
    # EWMA of bid-side vs ask-side event rates. Captures directional
    # clustering (Hawkes self-excitation). Centered at 0 (symmetric).
    # ------------------------------------------------------------------
    bid_ewma = ewma(is_bid, EWMA_ALPHA)
    ask_ewma = ewma(is_ask, EWMA_ALPHA)
    total_ewma = bid_ewma + ask_ewma
    total_ewma[total_ewma < 1e-8] = 1e-8
    # Ratio: 0.5 = balanced, >0.5 = more bids, <0.5 = more asks
    # Center at 0: (ratio - 0.5) * 2 → [-1, 1]
    new[:, 4] = ((bid_ewma / total_ewma - 0.5) * 2.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 20: realized_volatility (IC≈0.02)
    # Rolling sqrt(mean(price_change²)) over 200 events.
    # Regime conditioning variable: features behave differently in high vs low vol.
    # ------------------------------------------------------------------
    price_change = np.zeros(N, dtype=np.float64)
    price_change[1:] = np.diff(price_rel_ticks)
    price_change_sq = price_change ** 2
    rvol = np.sqrt(np.maximum(causal_rolling_mean(price_change_sq, 200), 0.0))
    new[:, 5] = rvol.astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 21: sweep_intensity (IC≈0.04)
    # Detects aggressive sweep patterns: 3+ consecutive same-side trades
    # within rapid succession. Sweeps are among the strongest directional signals.
    # Uses vectorized approach: count runs of same-side trades.
    # ------------------------------------------------------------------
    # Build signed trade indicator: +1 for ask-side trade, -1 for bid-side, 0 for non-trade
    trade_signed = is_trade * sign_side  # +1, -1, or 0

    # Count consecutive same-sign trades (vectorized run-length encoding)
    # For each position, how many consecutive same-sign trades precede it?
    same_as_prev = np.zeros(N, dtype=np.float64)
    same_as_prev[1:] = (trade_signed[1:] != 0) & (trade_signed[1:] == trade_signed[:-1]) & \
                        (time_delta_log[1:] < SWEEP_MAX_GAP_LOG)

    # Running consecutive count (reset on sign change or gap)
    consec = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        if same_as_prev[i]:
            consec[i] = consec[i - 1] + 1.0
        elif trade_signed[i] != 0:
            consec[i] = 1.0  # start of new potential run
        # else: consec[i] = 0 (non-trade event)

    # Sweep = run of 3+ same-side trades. Intensity = signed by direction.
    is_sweep = (consec >= SWEEP_MIN_TRADES).astype(np.float64)
    sweep_signed = is_sweep * trade_signed  # +1 buy sweep, -1 sell sweep

    # Rolling sum of sweep events over last 200 events
    new[:, 6] = causal_rolling_sum(sweep_signed, 200)

    return new


# ============================================================
# Smart per-feature normalization (original 15 + new 7)
# ============================================================
def apply_smart_normalization(ev_raw: np.ndarray, derived_v1: np.ndarray,
                               new_feats: np.ndarray) -> np.ndarray:
    """
    Apply smart per-feature treatment to all 22 features.
    Features 0-14: same as v1. Features 15-21: new v2 features.
    """
    N = ev_raw.shape[0]
    out = np.zeros((N, N_FEATURES), dtype=np.float32)

    # ── Original 15 features (same treatment as v1) ─────────────

    # 0: time_delta_log → clamp [0,8], /4.0
    out[:, 0] = np.clip(ev_raw[:, 0], 0.0, 8.0) / 4.0
    # 1: event_type_id → /4.0
    out[:, 1] = ev_raw[:, 1] / 4.0
    # 2: side_id → remap to -1/+1
    out[:, 2] = ev_raw[:, 2] * 2.0 - 1.0
    # 3: price_rel_ticks → /25.0
    out[:, 3] = np.clip(ev_raw[:, 3], -50.0, 50.0) / 25.0
    # 4: qty_log → (x - 0.693) / 3.0
    out[:, 4] = (ev_raw[:, 4] - 0.693) / 3.0
    # 5: spread_ticks → /5.0
    out[:, 5] = np.clip(ev_raw[:, 5], 0.0, 20.0) / 5.0
    # 6: cancel_side_asym_50 → /25.0
    out[:, 6] = np.clip(derived_v1[:, 0], -50.0, 50.0) / 25.0
    # 7: rolling_ofi_500 → rolling z-score
    out[:, 7] = causal_rolling_zscore(derived_v1[:, 1], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 8: event_density_20 → clamp [0,4], /2.0
    out[:, 8] = np.clip(derived_v1[:, 2], 0.0, 4.0) / 2.0
    # 9: price_mom_10 → rolling z-score
    out[:, 9] = causal_rolling_zscore(derived_v1[:, 3], ROLLING_ZSCORE_LOOKBACK_SHORT)
    # 10: qty_price_mom_50 → rolling z-score
    out[:, 10] = causal_rolling_zscore(derived_v1[:, 4], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 11: price_sign_momentum_200 → /100.0
    out[:, 11] = derived_v1[:, 5] / 100.0
    # 12: event_type_entropy_200 → /1.609
    out[:, 12] = derived_v1[:, 6] / 1.609
    # 13: fill_add_restoration_100 → as-is [0,1]
    out[:, 13] = derived_v1[:, 7]
    # 14: spread_velocity_50 → rolling z-score
    out[:, 14] = causal_rolling_zscore(derived_v1[:, 8], ROLLING_ZSCORE_LOOKBACK_SHORT)

    # ── New v2 features (15-21) ─────────────────────────────────

    # 15: queue_replenishment → rolling z-score (non-stationary ratio)
    out[:, 15] = causal_rolling_zscore(new_feats[:, 0], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 16: mom_divergence → already small scale [-1, 1]. Scale by *5 for model visibility.
    out[:, 16] = np.clip(new_feats[:, 1] * 5.0, -5.0, 5.0)

    # 17: ofi_x_spread → rolling z-score (non-stationary, scale varies with regime)
    out[:, 17] = causal_rolling_zscore(new_feats[:, 2], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 18: vol_weighted_pmom → rolling z-score (same as qty_price_mom)
    out[:, 18] = causal_rolling_zscore(new_feats[:, 3], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 19: buy_sell_intensity_ratio → already [-1, 1]. Keep as-is.
    out[:, 19] = new_feats[:, 4]

    # 20: realized_volatility → rolling z-score (regime-dependent)
    out[:, 20] = causal_rolling_zscore(new_feats[:, 5], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 21: sweep_intensity → /10.0 (bounded rolling sum, typical range [-20, 20])
    out[:, 21] = np.clip(new_feats[:, 6], -50.0, 50.0) / 10.0

    return out


# ============================================================
# Label keys to copy
# ============================================================
LABEL_KEYS = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]


# ============================================================
# Process a single file
# ============================================================
def process_file(src_path: Path, dst_path: Path) -> bool:
    """Load raw NPZ, compute all features, apply smart normalization, save."""
    try:
        t0 = time.time()
        d = np.load(src_path, allow_pickle=True)
        ev = d["events"].astype(np.float32)
        N = ev.shape[0]

        if ev.shape[1] != N_RAW:
            log(f"  SKIP {src_path.name}: unexpected shape {ev.shape}")
            d.close()
            return False

        if N < 1000:
            log(f"  SKIP {src_path.name}: too few events ({N:,})")
            d.close()
            return False

        # Step 1: Compute original 9 derived features
        derived_v1 = compute_derived_features_v1(ev)  # (N, 9)

        # Step 2: Compute 7 new v2 features
        new_feats = compute_new_features(ev)  # (N, 7)

        # Step 3: Apply smart normalization to all 22 features
        smart_features = apply_smart_normalization(ev, derived_v1, new_feats)  # (N, 22)

        # Step 4: Verify no NaN/Inf
        n_bad = np.count_nonzero(~np.isfinite(smart_features))
        if n_bad > 0:
            log(f"  WARNING: {n_bad} non-finite values in {src_path.name} — replacing with 0")
            smart_features = np.nan_to_num(smart_features, nan=0.0, posinf=5.0, neginf=-5.0)

        # Step 5: Save
        save_dict = {"events": smart_features}
        for key in LABEL_KEYS:
            if key in d:
                save_dict[key] = d[key]
        if "timestamps" in d:
            save_dict["timestamps"] = d["timestamps"]

        np.savez(dst_path, **save_dict)

        elapsed = time.time() - t0
        log(f"  OK {src_path.name}: {N:,} events -> {N_FEATURES} features ({elapsed:.1f}s)")
        d.close()
        return True

    except Exception as e:
        log(f"  ERROR {src_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================
def main():
    log(f"Smart Feature Preprocessing V2 (22 features)")
    log(f"  Input:  {INPUT_DIR}")
    log(f"  Output: {OUTPUT_DIR}")
    log(f"")
    log(f"  Original 15 features (0-14): same as v1")
    log(f"  NEW features:")
    log(f"    15 queue_replenishment     IC=0.074  rolling z-score")
    log(f"    16 mom_divergence          IC=0.066  *5 + clip")
    log(f"    17 ofi_x_spread            IC=0.053  rolling z-score")
    log(f"    18 vol_weighted_pmom       IC=0.053  rolling z-score")
    log(f"    19 buy_sell_intensity_ratio IC~0.03   as-is [-1,1]")
    log(f"    20 realized_volatility     IC~0.02   rolling z-score")
    log(f"    21 sweep_intensity         IC~0.04   /10.0")

    if not INPUT_DIR.exists():
        log(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_files = sorted(INPUT_DIR.glob("*.npz"))
    log(f"\n  Found {len(src_files)} input files")

    existing = {f.name for f in OUTPUT_DIR.glob("*.npz")}
    to_process = [(f, OUTPUT_DIR / f.name) for f in src_files if f.name not in existing]

    # Parallel worker support: WORKER_ID=0..N-1, WORKER_TOTAL=N
    worker_id = int(os.environ.get("WORKER_ID", 0))
    worker_total = int(os.environ.get("WORKER_TOTAL", 1))
    if worker_total > 1:
        to_process = [x for j, x in enumerate(to_process) if j % worker_total == worker_id]
        log(f"  Worker {worker_id}/{worker_total}: processing {len(to_process)} files")
    else:
        log(f"  Already processed: {len(existing)}, remaining: {len(to_process)}")

    if not to_process:
        log("  Nothing to do — all files already processed.")
        return

    n_ok, n_fail = 0, 0
    t_start = time.time()

    for i, (src, dst) in enumerate(to_process):
        log(f"\n  [{i+1}/{len(to_process)}] {src.name}...")
        if process_file(src, dst):
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.time() - t_start
    log(f"\n  Done: {n_ok} ok, {n_fail} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
