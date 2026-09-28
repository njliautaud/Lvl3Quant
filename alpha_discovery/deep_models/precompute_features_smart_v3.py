"""
precompute_features_smart_v3.py
===============================
Smart per-feature preprocessing V3 — 25 total features.

CHANGES FROM V2:
  - Feature 1 (event_type_id): NOW outputs 0.0/0.25/0.5/0.75/1.0 for model,
    BUT also saves raw integer event_type_id as separate array 'event_type_raw'
    so Mamba v7 can use a LEARNABLE EMBEDDING instead of this scalar.
  - 3 NEW features (22-24):
    22: ofi_short_100        — short-window OFI (100 events), rolling z-score
    23: ofi_long_2000        — long-window OFI (2000 events), rolling z-score
    24: ofi_acceleration     — rate of change of OFI (2nd derivative), rolling z-score
  - All v2 features (0-21) preserved exactly as-is for backward compatibility

Output: data/processed/mbo_events_smart_v3/ — (N, 25) float32 + event_type_raw (N,) int8
Training: MAMBA_FEATURE_SET=smart_v3, normalize_features=False

Usage:
    python precompute_features_smart_v3.py

    # Custom dirs:
    SMART_INPUT_DIR=... SMART_OUTPUT_DIR=... python precompute_features_smart_v3.py
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
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3"
))

# Rolling z-score lookback windows (in events)
ROLLING_ZSCORE_LOOKBACK_LONG = 10000
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000

# EWMA decay for intensity features
EWMA_ALPHA = 0.01  # ~100-event half-life

# Sweep detection parameters
SWEEP_MIN_TRADES = 3
SWEEP_MAX_GAP_LOG = 0.5

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
N_DERIVED_V1 = 9
N_V2_NEW = 7
N_V3_NEW = 3        # New features in v3
N_FEATURES = N_RAW + N_DERIVED_V1 + N_V2_NEW + N_V3_NEW  # 25


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
# Compute original 9 derived features (same as v1/v2)
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
# Compute v2 features (7 features, same as v2)
# ============================================================
def compute_v2_features(ev: np.ndarray) -> np.ndarray:
    """Compute 7 v2 features. Returns (N, 7)."""
    N = ev.shape[0]
    new = np.zeros((N, N_V2_NEW), dtype=np.float32)

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

    # 15: queue_replenishment
    add_rate = causal_rolling_sum(is_add * qty_log, 100)
    cancel_rate = causal_rolling_sum(is_cancel * qty_log, 100)
    new[:, 0] = (add_rate / np.maximum(cancel_rate, 0.1)).astype(np.float32)

    # 16: mom_divergence
    price_sign = np.sign(price_rel_ticks)
    pmom_short = causal_rolling_sum(price_sign, 20)
    pmom_long  = causal_rolling_sum(price_sign, 200)
    new[:, 1] = (pmom_short / 20.0 - pmom_long / 200.0).astype(np.float32)

    # 17: ofi_x_spread
    ofi_100 = causal_rolling_sum(qty_log * sign_side, 100)
    new[:, 2] = (ofi_100 * spread_ticks).astype(np.float32)

    # 18: vol_weighted_pmom
    new[:, 3] = causal_rolling_sum(qty_log * price_rel_ticks, 50)

    # 19: buy_sell_intensity_ratio
    bid_ewma = ewma(is_bid, EWMA_ALPHA)
    ask_ewma = ewma(is_ask, EWMA_ALPHA)
    total_ewma = bid_ewma + ask_ewma
    total_ewma[total_ewma < 1e-8] = 1e-8
    new[:, 4] = ((bid_ewma / total_ewma - 0.5) * 2.0).astype(np.float32)

    # 20: realized_volatility
    price_change = np.zeros(N, dtype=np.float64)
    price_change[1:] = np.diff(price_rel_ticks)
    price_change_sq = price_change ** 2
    rvol = np.sqrt(np.maximum(causal_rolling_mean(price_change_sq, 200), 0.0))
    new[:, 5] = rvol.astype(np.float32)

    # 21: sweep_intensity
    trade_signed = is_trade * sign_side
    same_as_prev = np.zeros(N, dtype=np.float64)
    same_as_prev[1:] = (trade_signed[1:] != 0) & (trade_signed[1:] == trade_signed[:-1]) & \
                        (time_delta_log[1:] < SWEEP_MAX_GAP_LOG)
    consec = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        if same_as_prev[i]:
            consec[i] = consec[i - 1] + 1.0
        elif trade_signed[i] != 0:
            consec[i] = 1.0
    is_sweep = (consec >= SWEEP_MIN_TRADES).astype(np.float64)
    sweep_signed = is_sweep * trade_signed
    new[:, 6] = causal_rolling_sum(sweep_signed, 200)

    return new


# ============================================================
# Compute NEW v3 features (3 additional)
# ============================================================
def compute_v3_features(ev: np.ndarray) -> np.ndarray:
    """
    Compute 3 new v3 features. Returns (N, 3).

    22: ofi_short_100     — short-timescale OFI (100 events). Captures immediate flow.
    23: ofi_long_2000     — long-timescale OFI (2000 events). Captures persistent flow.
    24: ofi_acceleration  — rate of change of OFI. 2nd derivative signal.
                           Computed as OFI_100 - OFI_500 (short minus medium).
                           Positive = flow accelerating in that direction.
    """
    N = ev.shape[0]
    v3 = np.zeros((N, N_V3_NEW), dtype=np.float32)

    side_id         = ev[:, 2]
    qty_log         = ev[:, 4].astype(np.float64)

    is_ask    = (side_id == _SIDE_ASK).astype(np.float64)
    is_bid    = (side_id == _SIDE_BID).astype(np.float64)
    sign_side = is_ask - is_bid

    ofi_signal = qty_log * sign_side

    # 22: ofi_short_100 — captures immediate aggressive flow
    v3[:, 0] = causal_rolling_sum(ofi_signal, 100)

    # 23: ofi_long_2000 — captures persistent directional pressure
    v3[:, 1] = causal_rolling_sum(ofi_signal, 2000)

    # 24: ofi_acceleration — 2nd derivative: how fast is flow changing?
    # OFI_100 - OFI_500 normalized by event count to same scale
    ofi_100_rate = causal_rolling_sum(ofi_signal, 100).astype(np.float64) / 100.0
    ofi_500_rate = causal_rolling_sum(ofi_signal, 500).astype(np.float64) / 500.0
    v3[:, 2] = (ofi_100_rate - ofi_500_rate).astype(np.float32)

    return v3


# ============================================================
# Smart per-feature normalization (all 25 features)
# ============================================================
def apply_smart_normalization(ev_raw: np.ndarray, derived_v1: np.ndarray,
                               v2_feats: np.ndarray, v3_feats: np.ndarray) -> np.ndarray:
    """
    Apply smart per-feature treatment to all 25 features.
    Features 0-14: same as v1/v2. Features 15-21: v2. Features 22-24: v3.
    """
    N = ev_raw.shape[0]
    out = np.zeros((N, N_FEATURES), dtype=np.float32)

    # ── Original 15 features (same treatment as v1/v2) ─────────────

    # 0: time_delta_log → clamp [0,8], /4.0
    out[:, 0] = np.clip(ev_raw[:, 0], 0.0, 8.0) / 4.0
    # 1: event_type_id → /4.0 (backward compat; Mamba v7 will use embedding instead)
    out[:, 1] = ev_raw[:, 1] / 4.0
    # 2: side_id → remap to -1/+1
    out[:, 2] = ev_raw[:, 2] * 2.0 - 1.0
    # 3: price_rel_ticks → clip and /25.0 (TICKS FROM MID — not absolute price)
    out[:, 3] = np.clip(ev_raw[:, 3], -50.0, 50.0) / 25.0
    # 4: qty_log → (x - 0.693) / 3.0
    out[:, 4] = (ev_raw[:, 4] - 0.693) / 3.0
    # 5: spread_ticks → clip /5.0
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

    # ── V2 features (15-21) — same as v2 ──────────────────────────

    # 15: queue_replenishment → rolling z-score
    out[:, 15] = causal_rolling_zscore(v2_feats[:, 0], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 16: mom_divergence → *5 + clip
    out[:, 16] = np.clip(v2_feats[:, 1] * 5.0, -5.0, 5.0)
    # 17: ofi_x_spread → rolling z-score
    out[:, 17] = causal_rolling_zscore(v2_feats[:, 2], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 18: vol_weighted_pmom → rolling z-score
    out[:, 18] = causal_rolling_zscore(v2_feats[:, 3], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 19: buy_sell_intensity_ratio → as-is [-1, 1]
    out[:, 19] = v2_feats[:, 4]
    # 20: realized_volatility → rolling z-score
    out[:, 20] = causal_rolling_zscore(v2_feats[:, 5], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 21: sweep_intensity → /10.0
    out[:, 21] = np.clip(v2_feats[:, 6], -50.0, 50.0) / 10.0

    # ── NEW V3 features (22-24) ────────────────────────────────────

    # 22: ofi_short_100 → rolling z-score (short-window flow)
    out[:, 22] = causal_rolling_zscore(v3_feats[:, 0], ROLLING_ZSCORE_LOOKBACK_SHORT)
    # 23: ofi_long_2000 → rolling z-score (persistent flow)
    out[:, 23] = causal_rolling_zscore(v3_feats[:, 1], ROLLING_ZSCORE_LOOKBACK_LONG)
    # 24: ofi_acceleration → rolling z-score (2nd derivative)
    out[:, 24] = causal_rolling_zscore(v3_feats[:, 2], ROLLING_ZSCORE_LOOKBACK_SHORT)

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
        derived_v1 = compute_derived_features_v1(ev)

        # Step 2: Compute 7 v2 features
        v2_feats = compute_v2_features(ev)

        # Step 3: Compute 3 v3 features
        v3_feats = compute_v3_features(ev)

        # Step 4: Apply smart normalization to all 25 features
        smart_features = apply_smart_normalization(ev, derived_v1, v2_feats, v3_feats)

        # Step 5: Verify no NaN/Inf
        n_bad = np.count_nonzero(~np.isfinite(smart_features))
        if n_bad > 0:
            log(f"  WARNING: {n_bad} non-finite values in {src_path.name} — replacing with 0")
            smart_features = np.nan_to_num(smart_features, nan=0.0, posinf=5.0, neginf=-5.0)

        # Step 6: Extract raw event_type_id for learnable embedding in Mamba v7
        event_type_raw = ev[:, 1].astype(np.int8)

        # Step 7: Save
        save_dict = {
            "events": smart_features,
            "event_type_raw": event_type_raw,
        }
        for key in LABEL_KEYS:
            if key in d:
                save_dict[key] = d[key]
        if "timestamps" in d:
            save_dict["timestamps"] = d["timestamps"]
        # Pass through raw MBO arrays for FIFO book reconstruction (if present)
        for raw_key in ("order_ids", "prices_raw", "sizes_raw", "sides_raw", "actions_raw"):
            if raw_key in d:
                save_dict[raw_key] = d[raw_key]

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
    log(f"Smart Feature Preprocessing V3 (25 features)")
    log(f"  Input:  {INPUT_DIR}")
    log(f"  Output: {OUTPUT_DIR}")
    log(f"")
    log(f"  Features 0-21: same as v2 (backward compatible)")
    log(f"  NEW features:")
    log(f"    22 ofi_short_100       — short-window OFI, rolling z-score")
    log(f"    23 ofi_long_2000       — long-window OFI, rolling z-score")
    log(f"    24 ofi_acceleration    — 2nd derivative of OFI, rolling z-score")
    log(f"  BONUS: event_type_raw saved for learnable embedding in Mamba v7")

    if not INPUT_DIR.exists():
        log(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_files = sorted(INPUT_DIR.glob("*.npz"))
    log(f"\n  Found {len(src_files)} input files")

    existing = {f.name for f in OUTPUT_DIR.glob("*.npz")}
    to_process = [(f, OUTPUT_DIR / f.name) for f in src_files if f.name not in existing]

    # Parallel worker support
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
