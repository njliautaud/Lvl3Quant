"""
precompute_book_normalized.py
=============================
Normalize raw book features (30 features from mbo_book_features) for neural network consumption.

Input:  data/processed/mbo_book_features/{date}_book_features.npz  (209 files, 30 raw features)
Output: data/processed/mbo_book_normalized/{date}_book_norm.npz

Output arrays:
    book_shape    (N, 20) float32 — Group A: spatial book structure (prices + sizes)
    book_dynamics (N, 10) float32 — Group B: temporal dynamics
    timestamps    (N,)    int64   — for alignment verification

Normalization strategy:
    Group A — Book Shape (spatial, for CNN):
        Prices [0:10]:  Convert to spread-relative distances from mid
        Sizes [10:20]:  log1p then causal rolling z-score (lookback=10000)

    Group B — Book Dynamics (temporal, for PatchTST):
        Feature-specific transforms (z-score, clip, log-sign, diff)

All outputs clipped to [-5, 5] as final safety.

Usage:
    python precompute_book_normalized.py
"""

import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

# ============================================================
# Config
# ============================================================
INPUT_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_book_features")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_book_normalized")

ROLLING_ZSCORE_LOOKBACK_LONG = 10000
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000
N_WORKERS = int(os.environ.get("BOOK_WORKERS", 1))


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ============================================================
# Causal rolling z-score (same pattern as smart_v3)
# ============================================================
def causal_rolling_zscore(signal: np.ndarray, W: int) -> np.ndarray:
    """Causal rolling z-score clipped to [-5, 5]. Cumsum-based O(N)."""
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


# ============================================================
# Process a single date
# ============================================================
def process_file(args):
    src_path, dst_path = args
    try:
        t0 = time.time()
        d = np.load(src_path)
        feat = d["features"]  # (N, 30) float32
        timestamps = d["timestamps"]  # (N,) int64
        N = feat.shape[0]

        if feat.shape[1] != 30:
            log(f"  SKIP {src_path.name}: unexpected shape {feat.shape}")
            return False

        if N < 1000:
            log(f"  SKIP {src_path.name}: too few events ({N:,})")
            return False

        # ============================================================
        # GROUP A: Book Shape (20 features) — spatial structure
        # ============================================================
        book_shape = np.zeros((N, 20), dtype=np.float32)

        # --- Prices [0:10]: spread-relative distances from mid ---
        bid_prices = feat[:, 0:5].astype(np.float64)   # bid_price_1..5
        ask_prices = feat[:, 5:10].astype(np.float64)   # ask_price_1..5

        mid = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0  # (N,)
        spread = ask_prices[:, 0] - bid_prices[:, 0]        # (N,)
        # Avoid division by zero — when spread is 0, use 1 tick
        spread_safe = np.where(spread > 0, spread, 1.0)

        # Price relative to mid, normalized by spread
        for i in range(5):
            book_shape[:, i] = ((bid_prices[:, i] - mid) / spread_safe).astype(np.float32)
            book_shape[:, 5 + i] = ((ask_prices[:, i] - mid) / spread_safe).astype(np.float32)

        # Handle early rows where book not yet built (prices = 0)
        # When a price level is 0, the book isn't populated there yet — set to 0
        for i in range(5):
            mask_bid = bid_prices[:, i] == 0
            mask_ask = ask_prices[:, i] == 0
            book_shape[mask_bid, i] = 0.0
            book_shape[mask_ask, 5 + i] = 0.0

        # --- Sizes [10:20]: log1p then rolling z-score ---
        bid_sizes = feat[:, 10:15].astype(np.float64)  # bid_size_1..5
        ask_sizes = feat[:, 15:20].astype(np.float64)  # ask_size_1..5

        for i in range(5):
            bid_log = np.log1p(bid_sizes[:, i]).astype(np.float32)
            ask_log = np.log1p(ask_sizes[:, i]).astype(np.float32)
            book_shape[:, 10 + i] = causal_rolling_zscore(bid_log, ROLLING_ZSCORE_LOOKBACK_LONG)
            book_shape[:, 15 + i] = causal_rolling_zscore(ask_log, ROLLING_ZSCORE_LOOKBACK_LONG)

        # ============================================================
        # GROUP B: Book Dynamics (10 features) — temporal signals
        # ============================================================
        book_dynamics = np.zeros((N, 10), dtype=np.float32)

        # [20] cum_delta → rolling z-score (lookback=10000)
        book_dynamics[:, 0] = causal_rolling_zscore(feat[:, 20], ROLLING_ZSCORE_LOOKBACK_LONG)

        # [21] rolling_imbalance_100 → rolling z-score (lookback=10000)
        book_dynamics[:, 1] = causal_rolling_zscore(feat[:, 21], ROLLING_ZSCORE_LOOKBACK_LONG)

        # [22] trade_intensity_100 → rolling z-score (lookback=5000)
        book_dynamics[:, 2] = causal_rolling_zscore(feat[:, 22], ROLLING_ZSCORE_LOOKBACK_SHORT)

        # [23] depth_imbalance_5 → already bounded [-1, 1], light z-score
        book_dynamics[:, 3] = causal_rolling_zscore(feat[:, 23], ROLLING_ZSCORE_LOOKBACK_LONG)

        # [24] spread_ticks → log1p(abs) * sign, then z-score
        spread_raw = feat[:, 24].astype(np.float64)
        spread_transformed = np.log1p(np.abs(spread_raw)) * np.sign(spread_raw)
        book_dynamics[:, 4] = causal_rolling_zscore(spread_transformed.astype(np.float32),
                                                     ROLLING_ZSCORE_LOOKBACK_LONG)

        # [25] bid_size_change → clip to [-50, 50] then /50
        book_dynamics[:, 5] = np.clip(feat[:, 25], -50.0, 50.0) / 50.0

        # [26] ask_size_change → clip to [-50, 50] then /50
        book_dynamics[:, 6] = np.clip(feat[:, 26], -50.0, 50.0) / 50.0

        # [27] mid_price_change_ticks → clip to [-5, 5] then /5
        book_dynamics[:, 7] = np.clip(feat[:, 27], -5.0, 5.0) / 5.0

        # [28] spread_change_ticks → clip to [-5, 5] then /5
        book_dynamics[:, 8] = np.clip(feat[:, 28], -5.0, 5.0) / 5.0

        # [29] net_order_flow → diff first (cumulative → incremental), then z-score
        nof = feat[:, 29].astype(np.float64)
        nof_diff = np.zeros(N, dtype=np.float32)
        nof_diff[1:] = np.diff(nof).astype(np.float32)
        book_dynamics[:, 9] = causal_rolling_zscore(nof_diff, ROLLING_ZSCORE_LOOKBACK_LONG)

        # ============================================================
        # Final safety: NaN → 0, Inf → clip, then clip all to [-5, 5]
        # ============================================================
        for arr in (book_shape, book_dynamics):
            n_bad = np.count_nonzero(~np.isfinite(arr))
            if n_bad > 0:
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
            arr[:] = np.clip(arr, -5.0, 5.0)

        # ============================================================
        # Save
        # ============================================================
        np.savez_compressed(dst_path,
                            book_shape=book_shape,
                            book_dynamics=book_dynamics,
                            timestamps=timestamps)

        elapsed = time.time() - t0
        log(f"  OK {src_path.name}: {N:>10,} events, shape=(20), dynamics=(10) [{elapsed:.1f}s]")
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
    log("Book Feature Normalization")
    log(f"  Input:  {INPUT_DIR}")
    log(f"  Output: {OUTPUT_DIR}")
    log(f"  Workers: {N_WORKERS}")
    log("")
    log("  Group A — Book Shape (20 features):")
    log("    [0:5]   bid_price_1..5  → spread-relative distance from mid")
    log("    [5:10]  ask_price_1..5  → spread-relative distance from mid")
    log("    [10:15] bid_size_1..5   → log1p + rolling z-score (W=10000)")
    log("    [15:20] ask_size_1..5   → log1p + rolling z-score (W=10000)")
    log("")
    log("  Group B — Book Dynamics (10 features):")
    log("    [0] cum_delta           → rolling z-score (W=10000)")
    log("    [1] rolling_imbalance   → rolling z-score (W=10000)")
    log("    [2] trade_intensity     → rolling z-score (W=5000)")
    log("    [3] depth_imbalance     → rolling z-score (W=10000)")
    log("    [4] spread_ticks        → log1p(abs)*sign + z-score (W=10000)")
    log("    [5] bid_size_change     → clip[-50,50]/50")
    log("    [6] ask_size_change     → clip[-50,50]/50")
    log("    [7] mid_price_change    → clip[-5,5]/5")
    log("    [8] spread_change       → clip[-5,5]/5")
    log("    [9] net_order_flow      → diff + z-score (W=10000)")

    if not INPUT_DIR.exists():
        log(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_files = sorted(INPUT_DIR.glob("*_book_features.npz"))
    log(f"\n  Found {len(src_files)} input files")

    # Check which are already done
    existing = set()
    for f in OUTPUT_DIR.glob("*_book_norm.npz"):
        existing.add(f.name)

    to_process = []
    for f in src_files:
        date_str = f.name.replace("_book_features.npz", "")
        out_name = f"{date_str}_book_norm.npz"
        if out_name not in existing:
            to_process.append((f, OUTPUT_DIR / out_name))

    log(f"  Already processed: {len(existing)}, remaining: {len(to_process)}")

    if not to_process:
        log("  Nothing to do — all files already processed.")
        return

    t_start = time.time()

    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            results = pool.map(process_file, to_process)
    else:
        results = [process_file(args) for args in to_process]

    n_ok = sum(1 for r in results if r)
    n_fail = sum(1 for r in results if not r)
    elapsed = time.time() - t_start

    log(f"\n  Done: {n_ok} ok, {n_fail} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
