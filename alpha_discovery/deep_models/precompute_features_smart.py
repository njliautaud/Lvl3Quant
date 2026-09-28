"""
precompute_features_smart.py
============================
Smart per-feature preprocessing for Mamba (and all event-driven models).

KEY INSIGHT: Blanket z-scoring hurts the model. Each feature needs treatment
that matches its nature:

  - Categorical (event_type, side) → encode meaningfully, DON'T z-score
  - Already bounded/log-scaled → fixed scaling, preserve signal
  - Non-stationary (OFI, momentum) → rolling z-score from PAST data only (causal)
  - Sparse features → address the sparsity, don't let 90% zeros dominate

PROBLEMS WITH CURRENT PER-FOLD Z-SCORE (what this fixes):
  1. event_type_id z-scored: imposes false ordinal relationship (type 2 ≠ "between" 1 and 3)
  2. side_id z-scored: 0/1 → (0-0.5)/0.5 = -1/+1 accidentally... but only by luck
  3. time_delta_log 89.5% zeros: z-score crushes the dynamic range of the 10.5% that matter
  4. rolling_ofi_500 (std=76) vs spread_velocity_50 (std=0.018): 4000x scale difference
     After z-score both are std≈1, but z-score from ENTIRE fold is slightly forward-looking
  5. Features like entropy [0, 1.6] and fill_add_ratio [0, 1] are already well-bounded

FEATURE TREATMENT GROUPS:

  Group 1 - Categorical (meaningful encoding, no normalization):
    1: event_type_id → /4.0 to [0, 1] (preserves ordinal-ish spacing for linear layer)
    2: side_id → remap to -1/+1 (directional signal, symmetric around zero)

  Group 2 - Bounded/log-scaled (fixed scaling, no rolling):
    0: time_delta_log → clamp [0, 8], /4.0 to [0, 2] (preserve actual timing signal)
    4: qty_log → (x - 0.693) / 3.0 to ~[0, 2.2] (center at min, scale nicely)
    5: spread_ticks → /5.0 to [0, 4] (raw spread level matters)
    12: event_type_entropy_200 → /1.609 to [0, 1] (already bounded)
    13: fill_add_restoration_100 → already [0, 1], leave as-is

  Group 3 - Bounded rolling features (fixed scaling by known bounds):
    3: price_rel_ticks → /25.0 to ~[-2, 2] (p5=-8.5, p95=8.5, so most in [-0.34, 0.34])
    6: cancel_side_asym_50 → /25.0 to ~[-2, 2]
    8: event_density_20 → clamp [0, 4], /2.0 to [0, 2] (like time_delta, sparse)
    11: price_sign_momentum_200 → /100.0 to [-2, 2]

  Group 4 - Non-stationary (CAUSAL rolling z-score, adapts to regime):
    7: rolling_ofi_500 → rolling z-score (lookback 10000 events)
    9: price_mom_10 → rolling z-score (lookback 5000 events)
    10: qty_price_mom_50 → rolling z-score (lookback 10000 events)
    14: spread_velocity_50 → rolling z-score (lookback 5000 events)

  Rolling z-score: at event i, mean/std computed from events [max(0, i-W):i].
  This is EXACTLY what would happen live — no future data, adapts to regime changes.

Output: data/processed/mbo_events_smart/ — (N, 15) float32, same feature count.
Can be used as drop-in replacement with normalize_features=False in the dataset.

Usage:
    python precompute_features_smart.py

    # Custom dirs:
    SMART_INPUT_DIR=path/to/mbo_events SMART_OUTPUT_DIR=path/to/mbo_events_smart python precompute_features_smart.py
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
    "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart"
))

# Rolling z-score lookback windows (in events)
ROLLING_ZSCORE_LOOKBACK_LONG = 10000   # For OFI, qty_price_mom (wider context)
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000   # For price_mom, spread_velocity

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
N_DERIVED = 9
N_FEATURES = N_RAW + N_DERIVED  # 15


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ============================================================
# Causal rolling helpers (same as precompute_features.py)
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
    """
    Causal rolling z-score: at event i, normalize by mean/std of past W events.
    z[i] = (signal[i] - mean(signal[max(0,i-W):i])) / (std(signal[max(0,i-W):i]) + eps)

    Uses running sum and sum-of-squares for O(N) computation.
    Clips output to [-5, 5] to prevent extreme values.
    """
    N = len(signal)
    sig64 = signal.astype(np.float64)

    # Running sums for mean
    cs = np.concatenate(([0.0], np.cumsum(sig64)))
    # Running sums-of-squares for variance
    cs_sq = np.concatenate(([0.0], np.cumsum(sig64 ** 2)))

    idx_end = np.arange(N, dtype=np.int64)
    idx_start = np.maximum(0, idx_end - W)
    counts = (idx_end - idx_start).astype(np.float64)
    counts[counts == 0] = 1.0

    sums = cs[idx_end] - cs[idx_start]
    sums_sq = cs_sq[idx_end] - cs_sq[idx_start]

    means = sums / counts
    variances = (sums_sq / counts) - (means ** 2)
    variances = np.maximum(variances, 0.0)  # numerical safety
    stds = np.sqrt(variances)
    stds[stds < 1e-8] = 1e-8  # prevent div-by-zero

    z = (sig64 - means) / stds

    # Clip to prevent extreme values
    z = np.clip(z, -5.0, 5.0)

    return z.astype(np.float32)


# ============================================================
# Compute 9 derived features (same logic as precompute_features.py)
# ============================================================
def compute_derived_features_raw(ev: np.ndarray) -> np.ndarray:
    """
    Compute 9 derived features and return the RAW (unnormalized) values.
    Returns (N, 9) array with features [6..14].
    """
    N = ev.shape[0]
    derived = np.zeros((N, N_DERIVED), dtype=np.float32)

    time_delta_log  = ev[:, 0].astype(np.float64)
    event_type_id   = ev[:, 1]
    side_id         = ev[:, 2]
    price_rel_ticks = ev[:, 3].astype(np.float64)
    qty_log         = ev[:, 4].astype(np.float64)
    spread_ticks    = ev[:, 5].astype(np.float64)

    is_cancel = (event_type_id == _EVENT_TYPE_CANCEL).astype(np.float64)
    is_ask    = (side_id == _SIDE_ASK).astype(np.float64)
    is_bid    = (side_id == _SIDE_BID).astype(np.float64)

    # Feature 6: cancel_side_asym_50
    cancel_signal = is_cancel * (is_ask - is_bid)
    derived[:, 0] = causal_rolling_sum(cancel_signal, 50)

    # Feature 7: rolling_ofi_500
    sign_side = is_ask - is_bid
    ofi_signal = qty_log * sign_side
    derived[:, 1] = causal_rolling_sum(ofi_signal, 500)

    # Feature 8: event_density_20
    derived[:, 2] = causal_rolling_mean(time_delta_log, 20)

    # Feature 9: price_mom_10
    derived[:, 3] = causal_rolling_sum(price_rel_ticks, 10)

    # Feature 10: qty_price_mom_50
    qty_price = qty_log * price_rel_ticks
    derived[:, 4] = causal_rolling_sum(qty_price, 50)

    # Feature 11: price_sign_momentum_200
    price_sign = np.sign(price_rel_ticks)
    derived[:, 5] = causal_rolling_sum(price_sign, 200)

    # Feature 12: event_type_entropy_200
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

    # Feature 13: fill_add_restoration_100
    is_fill_trade = ((event_type_id == _EVENT_TYPE_FILL) |
                     (event_type_id == _EVENT_TYPE_TRADE)).astype(np.float64)
    is_add = (event_type_id == _EVENT_TYPE_ADD).astype(np.float64)
    prev_is_fill_trade = np.empty(N, dtype=np.float64)
    prev_is_fill_trade[0] = 0.0
    prev_is_fill_trade[1:] = is_fill_trade[:-1]
    prev_side = np.empty(N, dtype=np.float64)
    prev_side[0] = -1.0
    prev_side[1:] = side_id[:-1].astype(np.float64)
    same_side = (side_id.astype(np.float64) == prev_side).astype(np.float64)
    fill_add_signal = prev_is_fill_trade * is_add * same_side

    W_FAR = 100
    fill_add_cs   = np.concatenate(([0.0], np.cumsum(fill_add_signal)))
    fill_trade_cs = np.concatenate(([0.0], np.cumsum(is_fill_trade)))
    idx_end_far   = np.arange(N, dtype=np.int64)
    idx_start_far = np.maximum(0, idx_end_far - W_FAR)
    pair_counts  = fill_add_cs[idx_end_far]   - fill_add_cs[idx_start_far]
    fill_counts  = fill_trade_cs[idx_end_far] - fill_trade_cs[idx_start_far]
    fill_counts  = np.maximum(fill_counts, 1.0)
    derived[:, 7] = (pair_counts / fill_counts).astype(np.float32)

    # Feature 14: spread_velocity_50
    spread_diff = np.empty(N, dtype=np.float64)
    spread_diff[0] = 0.0
    spread_diff[1:] = spread_ticks[1:] - spread_ticks[:-1]
    derived[:, 8] = causal_rolling_mean(spread_diff, 50)

    return derived


# ============================================================
# Smart per-feature normalization
# ============================================================
def apply_smart_normalization(ev_raw: np.ndarray, derived_raw: np.ndarray) -> np.ndarray:
    """
    Apply smart per-feature treatment to raw 6 features + 9 derived features.

    Returns (N, 15) array ready for model consumption.
    """
    N = ev_raw.shape[0]
    out = np.zeros((N, N_FEATURES), dtype=np.float32)

    # ── Group 1: Categorical (meaningful encoding) ──────────────

    # 0: time_delta_log — already log-scaled. Clamp and scale.
    #    89.5% zeros (simultaneous events). Don't z-score — the raw value IS the signal.
    #    Clamp [0, 8] then /4.0 → [0, 2]. Most values near 0, spikes up to 2.
    out[:, 0] = np.clip(ev_raw[:, 0], 0.0, 8.0) / 4.0

    # 1: event_type_id — categorical 0-4. Scale to [0, 1].
    #    Model learns that 0→0, 1→0.25, 2→0.5, 3→0.75, 4→1.0
    #    Not perfect (imposes ordering) but much better than z-scoring
    out[:, 1] = ev_raw[:, 1] / 4.0

    # 2: side_id — binary. Remap to -1/+1 (symmetric directional signal).
    #    bid(0)→-1, ask(1)→+1. Centered at zero, symmetric magnitude.
    out[:, 2] = ev_raw[:, 2] * 2.0 - 1.0

    # ── Group 2: Bounded/log-scaled (fixed scaling) ─────────────

    # 3: price_rel_ticks — clipped [-50, 50]. Scale by /25 → ~[-2, 2].
    #    P5=-8.5, P95=8.5, so 90% of data in [-0.34, 0.34]. Tails to ±2.
    out[:, 3] = np.clip(ev_raw[:, 3], -50.0, 50.0) / 25.0

    # 4: qty_log — [0.693, 7.33]. Center at minimum, scale.
    #    (x - 0.693) / 3.0 → [0, 2.2]. Zero = 1 lot (minimum), 2.2 = huge order.
    out[:, 4] = (ev_raw[:, 4] - 0.693) / 3.0

    # 5: spread_ticks — integer [0, 20]. /5.0 → [0, 4].
    #    56% zero (inside spread or at NBBO). The raw level is informative.
    out[:, 5] = np.clip(ev_raw[:, 5], 0.0, 20.0) / 5.0

    # 6: cancel_side_asym_50 — bounded [-50, 50]. /25.0 → [-2, 2].
    out[:, 6] = np.clip(derived_raw[:, 0], -50.0, 50.0) / 25.0

    # ── Group 3: Non-stationary (causal rolling z-score) ────────

    # 7: rolling_ofi_500 — std=76.6, range [-1116, 764]. Regime-dependent.
    #    Rolling z-score with 10000-event lookback. Exactly like live.
    out[:, 7] = causal_rolling_zscore(derived_raw[:, 1], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 8: event_density_20 — like time_delta_log (85.6% zero). Fixed scale.
    #    Clamp [0, 4], /2.0 → [0, 2].
    out[:, 8] = np.clip(derived_raw[:, 2], 0.0, 4.0) / 2.0

    # 9: price_mom_10 — rolling sum, regime-dependent.
    #    Rolling z-score with 5000-event lookback.
    out[:, 9] = causal_rolling_zscore(derived_raw[:, 3], ROLLING_ZSCORE_LOOKBACK_SHORT)

    # 10: qty_price_mom_50 — large scale, regime-dependent.
    #    Rolling z-score with 10000-event lookback.
    out[:, 10] = causal_rolling_zscore(derived_raw[:, 4], ROLLING_ZSCORE_LOOKBACK_LONG)

    # 11: price_sign_momentum_200 — bounded [-200, 200]. Fixed scale.
    #    /100.0 → [-2, 2].
    out[:, 11] = derived_raw[:, 5] / 100.0

    # 12: event_type_entropy_200 — bounded [0, ln(5)≈1.609]. /1.609 → [0, 1].
    out[:, 12] = derived_raw[:, 6] / 1.609

    # 13: fill_add_restoration_100 — already [0, 1]. Keep as-is.
    out[:, 13] = derived_raw[:, 7]

    # 14: spread_velocity_50 — tiny scale (std=0.018). Rolling z-score.
    out[:, 14] = causal_rolling_zscore(derived_raw[:, 8], ROLLING_ZSCORE_LOOKBACK_SHORT)

    return out


# ============================================================
# Label keys to copy
# ============================================================
LABEL_KEYS = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]


# ============================================================
# Process a single file
# ============================================================
def process_file(src_path: Path, dst_path: Path) -> bool:
    """Load raw NPZ, compute derived features, apply smart normalization, save."""
    try:
        t0 = time.time()
        d = np.load(src_path, allow_pickle=True)
        ev = d["events"].astype(np.float32)
        N = ev.shape[0]

        if ev.shape[1] != N_RAW:
            log(f"  SKIP {src_path.name}: unexpected shape {ev.shape} (expected N x {N_RAW})")
            d.close()
            return False

        if N < 1000:
            log(f"  SKIP {src_path.name}: too few events ({N:,}) — likely bad/partial day")
            d.close()
            return False

        # Step 1: Compute 9 derived features (raw, unnormalized)
        derived_raw = compute_derived_features_raw(ev)  # (N, 9)

        # Step 2: Apply smart per-feature normalization
        smart_features = apply_smart_normalization(ev, derived_raw)  # (N, 15)

        # Step 3: Verify no NaN/Inf
        n_bad = np.count_nonzero(~np.isfinite(smart_features))
        if n_bad > 0:
            log(f"  WARNING: {n_bad} non-finite values in {src_path.name} — replacing with 0")
            smart_features = np.nan_to_num(smart_features, nan=0.0, posinf=5.0, neginf=-5.0)

        # Step 4: Build output dict
        save_dict = {"events": smart_features}

        # Copy labels and timestamps
        for key in LABEL_KEYS:
            if key in d:
                save_dict[key] = d[key]
        if "timestamps" in d:
            save_dict["timestamps"] = d["timestamps"]

        # Save
        np.savez(dst_path, **save_dict)

        elapsed = time.time() - t0
        log(f"  ✓ {src_path.name}: {N:,} events → {dst_path.name} ({elapsed:.1f}s)")
        d.close()
        return True

    except Exception as e:
        log(f"  ERROR {src_path.name}: {e}")
        return False


# ============================================================
# Main
# ============================================================
def main():
    log(f"Smart Feature Preprocessing")
    log(f"  Input:  {INPUT_DIR}")
    log(f"  Output: {OUTPUT_DIR}")
    log(f"  Rolling z-score lookback: {ROLLING_ZSCORE_LOOKBACK_LONG} (long), {ROLLING_ZSCORE_LOOKBACK_SHORT} (short)")
    log(f"")
    log(f"  Feature treatment:")
    log(f"    0  time_delta_log        → clamp [0,8], /4.0")
    log(f"    1  event_type_id         → /4.0 to [0,1]")
    log(f"    2  side_id               → remap to -1/+1")
    log(f"    3  price_rel_ticks       → /25.0")
    log(f"    4  qty_log               → (x-0.693)/3.0")
    log(f"    5  spread_ticks          → /5.0")
    log(f"    6  cancel_side_asym_50   → /25.0")
    log(f"    7  rolling_ofi_500       → ROLLING z-score (10K lookback)")
    log(f"    8  event_density_20      → clamp [0,4], /2.0")
    log(f"    9  price_mom_10          → ROLLING z-score (5K lookback)")
    log(f"    10 qty_price_mom_50      → ROLLING z-score (10K lookback)")
    log(f"    11 price_sign_momentum   → /100.0")
    log(f"    12 event_type_entropy    → /1.609")
    log(f"    13 fill_add_restoration  → as-is [0,1]")
    log(f"    14 spread_velocity_50    → ROLLING z-score (5K lookback)")

    if not INPUT_DIR.exists():
        log(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all NPZ files
    src_files = sorted(INPUT_DIR.glob("*.npz"))
    log(f"\n  Found {len(src_files)} input files")

    # Skip already-processed
    existing = {f.name for f in OUTPUT_DIR.glob("*.npz")}
    to_process = [(f, OUTPUT_DIR / f.name) for f in src_files if f.name not in existing]
    log(f"  Already processed: {len(existing)}, remaining: {len(to_process)}")

    if not to_process:
        log("  Nothing to do — all files already processed.")
        return

    n_ok, n_fail = 0, 0
    t_start = time.time()

    for i, (src, dst) in enumerate(to_process):
        log(f"\n  [{i+1}/{len(to_process)}] Processing {src.name}...")
        if process_file(src, dst):
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.time() - t_start
    log(f"\n  Done: {n_ok} ok, {n_fail} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
