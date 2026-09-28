"""
precompute_features.py
======================
Precomputes 9 derived features for all MBO event NPZ files and saves
augmented (N, 15) NPZ files to a separate output directory.

This eliminates memory spikes caused by on-the-fly feature computation
during CNN training (rolling window cumsum ops on large arrays).

Input:  data/processed/mbo_events/       — raw (N, 6) events arrays
Output: data/processed/mbo_events_feat/  — augmented (N, 15) events arrays

Usage:
    python precompute_features.py

    # Custom dirs via env vars:
    CNN_DATA_DIR=... CNN_FEAT_DIR=... python precompute_features.py

Feature columns in output 'events' array (N, 15):
    0:  time_delta_log
    1:  event_type_id
    2:  side_id
    3:  price_rel_ticks
    4:  qty_log
    5:  spread_ticks
    6:  cancel_side_asym_50
    7:  rolling_ofi_500
    8:  event_density_20
    9:  price_mom_10
    10: qty_price_mom_50
    11: price_sign_momentum_200
    12: event_type_entropy_200
    13: fill_add_restoration_100
    14: spread_velocity_50

Resume support: files already present in the output directory are skipped.
Chunked processing: files with >5M events are processed 1M events at a time
to avoid RAM spikes from large intermediate arrays.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# Config
# ============================================================
DATA_DIR = Path(os.environ.get(
    "CNN_DATA_DIR",
    "C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/mbo_events"
))
FEAT_DIR = Path(os.environ.get(
    "CNN_FEAT_DIR",
    "C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/mbo_events_feat"
))

CHUNK_SIZE = int(os.environ.get("PRECOMPUTE_CHUNK_SIZE", 1_000_000))  # events per chunk
LARGE_FILE_THRESHOLD = int(os.environ.get("PRECOMPUTE_LARGE_THRESHOLD", 5_000_000))

# Encoder conventions — must match train_cnn_clean_s84.py
_EVENT_TYPE_ADD    = int(os.environ.get("CNN_ADD_CODE",      "0"))
_EVENT_TYPE_CANCEL = int(os.environ.get("CNN_CANCEL_CODE",   "1"))
_EVENT_TYPE_MODIFY = int(os.environ.get("CNN_MODIFY_CODE",   "2"))
_EVENT_TYPE_TRADE  = int(os.environ.get("CNN_TRADE_CODE",    "3"))
_EVENT_TYPE_FILL   = int(os.environ.get("CNN_FILL_CODE",     "4"))
_SIDE_BID          = int(os.environ.get("CNN_BID_CODE",      "0"))
_SIDE_ASK          = int(os.environ.get("CNN_ASK_CODE",      "1"))
_N_EVENT_TYPES     = int(os.environ.get("CNN_N_EVENT_TYPES", "5"))

N_RAW_FEATURES = 6
N_DERIVED      = 9
N_FEATURES     = N_RAW_FEATURES + N_DERIVED  # 15


# ============================================================
# Logging
# ============================================================
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ============================================================
# Core feature computation — copied EXACTLY from train_cnn_clean_s84.py
# ============================================================
def compute_derived_features(ev: np.ndarray) -> np.ndarray:
    """
    Given raw events array of shape (N, 6), compute 9 causal derived features
    and return shape (N, 15).

    All rolling windows are implemented with cumsum shifts — O(N), fully causal.
    NaN-fill strategy: fill first W events with 0.0 (neutral) so no NaNs propagate
    into the model. Clipping to [-10, 10] happens after normalization in _load_day.

    Features 6-14:
      6:  cancel_side_asym_50        — rolling ask-cancel minus bid-cancel over 50 events
      7:  rolling_ofi_500            — order flow imbalance over 500 events
      8:  event_density_20           — rolling mean of time_delta_log over 20 events
      9:  price_mom_10               — rolling sum of price_rel_ticks over 10 events
      10: qty_price_mom_50           — rolling sum of qty_log * price_rel_ticks over 50 events
      11: price_sign_momentum_200    — rolling sum of sign(price_rel_ticks) over 200 events
      12: event_type_entropy_200     — Shannon entropy of event_type_id over 200-event window
      13: fill_add_restoration_100   — fill-then-add pair ratio over 100 events
      14: spread_velocity_50         — rolling mean of spread first-difference over 50 events
    """
    N = ev.shape[0]
    out = np.empty((N, N_RAW_FEATURES + N_DERIVED), dtype=np.float32)
    out[:, :N_RAW_FEATURES] = ev

    time_delta_log  = ev[:, 0].astype(np.float64)
    event_type_id   = ev[:, 1]
    side_id         = ev[:, 2]
    price_rel_ticks = ev[:, 3].astype(np.float64)
    qty_log         = ev[:, 4].astype(np.float64)

    # ------------------------------------------------------------------
    # Helper: causal rolling sum over window W using cumsum.
    # result[i] = sum(signal[i-W : i])  (excludes i itself for strict causality)
    # For i < W, use the available history sum(signal[0 : i]).
    # ------------------------------------------------------------------
    def causal_rolling_sum(signal: np.ndarray, W: int) -> np.ndarray:
        cs = np.concatenate(([0.0], np.cumsum(signal)))  # length N+1
        # result[i] = cs[i] - cs[max(0, i-W)]
        # Using cs[i] (not cs[i+1]) means we EXCLUDE event i itself → fully causal.
        idx_end   = np.arange(N, dtype=np.int64)          # [0, 1, ..., N-1]
        idx_start = np.maximum(0, idx_end - W)             # [max(0, i-W)]
        return (cs[idx_end] - cs[idx_start]).astype(np.float32)

    def causal_rolling_mean(signal: np.ndarray, W: int) -> np.ndarray:
        cs = np.concatenate(([0.0], np.cumsum(signal)))
        idx_end   = np.arange(N, dtype=np.int64)
        idx_start = np.maximum(0, idx_end - W)
        counts    = (idx_end - idx_start).astype(np.float32)
        counts[counts == 0] = 1.0  # avoid div-by-zero at i=0
        return ((cs[idx_end] - cs[idx_start]) / counts).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 6: cancel_side_asym_50
    # Signal per event: +1 if Ask Cancel, -1 if Bid Cancel, 0 otherwise.
    # Rolling sum over 50 past events.
    # ------------------------------------------------------------------
    is_cancel = (event_type_id == _EVENT_TYPE_CANCEL).astype(np.float64)
    is_ask    = (side_id == _SIDE_ASK).astype(np.float64)
    is_bid    = (side_id == _SIDE_BID).astype(np.float64)
    cancel_signal = is_cancel * (is_ask - is_bid)   # +1, -1, or 0
    out[:, 6] = causal_rolling_sum(cancel_signal, 50)

    # ------------------------------------------------------------------
    # Feature 7: rolling_ofi_500
    # OFI per event: qty_log * sign_side (+1 buy / -1 sell).
    # Rolling sum over 500 past events.
    # ------------------------------------------------------------------
    sign_side = is_ask - is_bid   # +1 for ask-initiated (buy), -1 for bid (sell)
    ofi_signal = qty_log * sign_side
    out[:, 7] = causal_rolling_sum(ofi_signal, 500)

    # ------------------------------------------------------------------
    # Feature 8: event_density_20
    # Rolling mean of time_delta_log over 20 past events.
    # High value = slow (sparse), low value = fast (dense).
    # ------------------------------------------------------------------
    out[:, 8] = causal_rolling_mean(time_delta_log, 20)

    # ------------------------------------------------------------------
    # Feature 9: price_mom_10
    # Rolling sum of price_rel_ticks over 10 past events.
    # ------------------------------------------------------------------
    out[:, 9] = causal_rolling_sum(price_rel_ticks, 10)

    # ------------------------------------------------------------------
    # Feature 10: qty_price_mom_50
    # Rolling sum of (qty_log * price_rel_ticks) over 50 past events.
    # Captures signed volume momentum.
    # ------------------------------------------------------------------
    qty_price = qty_log * price_rel_ticks
    out[:, 10] = causal_rolling_sum(qty_price, 50)

    # ------------------------------------------------------------------
    # Feature 11: price_sign_momentum_200
    # Rolling sum of sign(price_rel_ticks) over 200 past events.
    # sign encodes direction (+1/-1/0), stripping magnitude noise.
    # IC=+0.060 (strongest new feature).
    # ------------------------------------------------------------------
    price_sign = np.sign(price_rel_ticks)
    out[:, 11] = causal_rolling_sum(price_sign, 200)

    # ------------------------------------------------------------------
    # Feature 12: event_type_entropy_200
    # Shannon entropy of event_type_id distribution over a causal 200-event window.
    # H = -sum(p_k * log(p_k)) for k in event types present in window.
    # Low entropy = dominated by one type; high entropy = mixed activity.
    # IC=-0.028.
    #
    # Implementation: maintain per-type cumsum arrays, then compute histogram
    # counts for each position using the same cumsum-shift trick. O(N * K).
    # ------------------------------------------------------------------
    K = _N_EVENT_TYPES
    W_ENT = 200
    type_cs = np.zeros((K, N + 1), dtype=np.float64)
    for k in range(K):
        indicator = (event_type_id == k).astype(np.float64)
        type_cs[k] = np.concatenate(([0.0], np.cumsum(indicator)))

    idx_end_ent   = np.arange(N, dtype=np.int64)
    idx_start_ent = np.maximum(0, idx_end_ent - W_ENT)
    window_counts = (idx_end_ent - idx_start_ent).astype(np.float64)
    window_counts[window_counts == 0] = 1.0            # avoid div-by-zero at i=0

    entropy = np.zeros(N, dtype=np.float32)
    for k in range(K):
        counts_k = type_cs[k][idx_end_ent] - type_cs[k][idx_start_ent]
        p_k = counts_k / window_counts
        # -p * log(p) with safe handling of p=0 (contribution is 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(p_k > 0.0, np.log(p_k), 0.0)
        entropy -= (p_k * log_p).astype(np.float32)
    out[:, 12] = entropy

    # ------------------------------------------------------------------
    # Feature 13: fill_add_restoration_100
    # For each event i, look back over the 100-event causal window and count
    # "fill-then-add" consecutive pairs: event at position j is a Fill/Trade AND
    # event at j+1 is an Add on the same side.
    # Feature = (count of such pairs) / max(1, count of fills in window).
    # IC=+0.015.
    #
    # Implementation: build a binary "fill_then_add_same_side" signal per event,
    # where signal[i] = 1 if event[i-1] is fill/trade AND event[i] is add on same side.
    # Then causal rolling sum of signal / causal rolling sum of is_fill_trade.
    # ------------------------------------------------------------------
    is_fill_trade = ((event_type_id == _EVENT_TYPE_FILL) |
                     (event_type_id == _EVENT_TYPE_TRADE)).astype(np.float64)
    is_add = (event_type_id == _EVENT_TYPE_ADD).astype(np.float64)

    # prev_is_fill_trade[i] = is_fill_trade[i-1]
    prev_is_fill_trade = np.empty(N, dtype=np.float64)
    prev_is_fill_trade[0] = 0.0
    prev_is_fill_trade[1:] = is_fill_trade[:-1]

    # same_side[i] = 1 if side_id[i] == side_id[i-1], else 0
    prev_side = np.empty(N, dtype=np.float64)
    prev_side[0] = -1.0  # neutral sentinel, never matches a real side_id
    prev_side[1:] = side_id[:-1].astype(np.float64)
    same_side = (side_id.astype(np.float64) == prev_side).astype(np.float64)

    fill_add_signal = prev_is_fill_trade * is_add * same_side  # 1 when pattern matches

    W_FAR = 100
    fill_add_cs   = np.concatenate(([0.0], np.cumsum(fill_add_signal)))
    fill_trade_cs = np.concatenate(([0.0], np.cumsum(is_fill_trade)))

    idx_end_far   = np.arange(N, dtype=np.int64)
    idx_start_far = np.maximum(0, idx_end_far - W_FAR)

    pair_counts  = fill_add_cs[idx_end_far]   - fill_add_cs[idx_start_far]
    fill_counts  = fill_trade_cs[idx_end_far] - fill_trade_cs[idx_start_far]
    fill_counts  = np.maximum(fill_counts, 1.0)  # avoid div-by-zero
    out[:, 13] = (pair_counts / fill_counts).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 14: spread_velocity_50
    # Causal rolling mean of the first-difference of spread_ticks over 50 events.
    # spread_diff[i] = spread_ticks[i] - spread_ticks[i-1] (0 at i=0).
    # Captures whether spread is widening or narrowing.
    # IC=+0.012.
    # ------------------------------------------------------------------
    spread_ticks_col = ev[:, 5].astype(np.float64)
    spread_diff = np.empty(N, dtype=np.float64)
    spread_diff[0] = 0.0
    spread_diff[1:] = spread_ticks_col[1:] - spread_ticks_col[:-1]
    out[:, 14] = causal_rolling_mean(spread_diff, 50)

    return out


def compute_derived_features_chunked(ev: np.ndarray) -> np.ndarray:
    """
    Process a large events array in chunks of CHUNK_SIZE to avoid RAM spikes.

    The rolling features use a window of at most 500 events, so we overlap
    chunks by max_window (500) events to ensure correct boundary values.
    We discard the overlap prefix from each chunk's output.
    """
    N = ev.shape[0]
    MAX_WINDOW = 500  # largest rolling window used in any feature

    if N <= CHUNK_SIZE:
        return compute_derived_features(ev)

    log(f"    Chunked processing: {N:,} events, chunk_size={CHUNK_SIZE:,}, overlap={MAX_WINDOW}")

    out = np.empty((N, N_FEATURES), dtype=np.float32)
    cursor = 0  # write position in out

    chunk_start = 0
    while chunk_start < N:
        chunk_end = min(chunk_start + CHUNK_SIZE, N)

        # Include overlap prefix from prior chunk for correct rolling values
        overlap_start = max(0, chunk_start - MAX_WINDOW)
        chunk_ev = ev[overlap_start:chunk_end]

        chunk_out = compute_derived_features(chunk_ev)

        # Discard the overlap prefix — those events were already written
        keep_from = chunk_start - overlap_start  # how many rows to drop at front
        valid_out = chunk_out[keep_from:]        # rows that map to [chunk_start, chunk_end)

        n_valid = valid_out.shape[0]
        out[cursor:cursor + n_valid] = valid_out
        cursor += n_valid

        log(f"    Chunk [{chunk_start:,}:{chunk_end:,}] done, wrote {n_valid:,} rows")
        chunk_start = chunk_end

    assert cursor == N, f"Chunked output size mismatch: {cursor} != {N}"
    return out


# ============================================================
# Label key names to copy verbatim from input NPZ
# ============================================================
LABEL_KEYS = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]


def process_file(src_path: Path, dst_path: Path) -> bool:
    """
    Load src NPZ, compute derived features, save augmented NPZ to dst.
    Returns True on success, False on error.
    """
    try:
        t0 = time.time()
        d = np.load(src_path, allow_pickle=True)
        ev = d["events"].astype(np.float32)
        N = ev.shape[0]

        if ev.shape[1] != N_RAW_FEATURES:
            log(f"  SKIP {src_path.name}: unexpected shape {ev.shape} (expected N x {N_RAW_FEATURES})")
            d.close()
            return False

        # Choose flat or chunked processing
        if N > LARGE_FILE_THRESHOLD:
            log(f"  Large file ({N:,} events) — using chunked processing")
            ev_feat = compute_derived_features_chunked(ev)
        else:
            ev_feat = compute_derived_features(ev)

        # Collect label arrays (copy whatever keys exist)
        save_dict = {"events": ev_feat}
        for key in LABEL_KEYS:
            if key in d:
                save_dict[key] = d[key]

        d.close()

        np.savez_compressed(dst_path, **save_dict)
        elapsed = time.time() - t0
        log(f"  Saved {dst_path.name} ({N:,} events, {ev_feat.shape[1]} features, {elapsed:.1f}s)")
        return True

    except Exception as e:
        log(f"  ERROR processing {src_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("precompute_features.py — MBO event feature precomputation")
    log("=" * 60)
    log(f"Input dir:  {DATA_DIR}")
    log(f"Output dir: {FEAT_DIR}")
    log(f"Chunk size: {CHUNK_SIZE:,} events")
    log(f"Large file threshold: {LARGE_FILE_THRESHOLD:,} events")
    log(f"Event codes: Add={_EVENT_TYPE_ADD} Cancel={_EVENT_TYPE_CANCEL} "
        f"Modify={_EVENT_TYPE_MODIFY} Trade={_EVENT_TYPE_TRADE} Fill={_EVENT_TYPE_FILL}")
    log(f"Side codes: Bid={_SIDE_BID} Ask={_SIDE_ASK}")

    if not DATA_DIR.exists():
        log(f"ERROR: Input directory does not exist: {DATA_DIR}")
        sys.exit(1)

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all NPZ files, sort for deterministic ordering
    all_files = sorted(DATA_DIR.glob("*.npz"))
    if not all_files:
        log(f"ERROR: No .npz files found in {DATA_DIR}")
        sys.exit(1)

    log(f"Found {len(all_files)} NPZ files in input directory")

    # Determine which files need processing (resume support)
    to_process = []
    skipped = 0
    for src in all_files:
        dst = FEAT_DIR / src.name
        if dst.exists():
            skipped += 1
        else:
            to_process.append(src)

    log(f"Already done: {skipped} files | To process: {len(to_process)} files")

    if not to_process:
        log("All files already precomputed. Nothing to do.")
        return

    # Process files
    n_ok = 0
    n_err = 0
    t_total = time.time()

    for i, src in enumerate(to_process, 1):
        dst = FEAT_DIR / src.name
        log(f"[{i}/{len(to_process)}] {src.name}")
        ok = process_file(src, dst)
        if ok:
            n_ok += 1
        else:
            n_err += 1

    elapsed = time.time() - t_total
    log("=" * 60)
    log(f"Done. {n_ok} succeeded, {n_err} failed, {skipped} skipped. "
        f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Output: {FEAT_DIR}")

    if n_err > 0:
        log(f"WARNING: {n_err} files failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
