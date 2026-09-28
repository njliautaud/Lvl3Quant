"""
build_confluence_features.py
============================
Build enriched training data combining CNN-Mamba v2 + PatchTST predictions with
market microstructure features for execution model training.

Key facts discovered during development:
  - CNN-Mamba: WINDOW_SIZE=3000, STRIDE=500 (auto-detected), outputs predictions[] (N, 3)
  - PatchTST:  WINDOW_SIZE=500,  STRIDE=250 (auto-detected), outputs predictions[] (N, 3)
  - Both NPZ formats: {'predictions':(N,3), 'labels':(N,3), 'horizons':(3,), 'oot_files':(1,)}
  - NO timestamps in prediction NPZs — must reconstruct from MBO data using stride logic
  - exec_features_v1: 44-feature windows, DECISION_STRIDE=5000 per day
  - MBO data: /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/YYYYMMDD_mbo_events.npz
  - Cost: 0.376t commission only (NO separate spread cost for FIFO fills)

Output per date:
  /home/jupiter/Lvl3Quant/output/confluence_features/{date}_confluence.npz

Combined:
  /home/jupiter/Lvl3Quant/output/confluence_features/all_confluence_features.npz

Usage:
  python build_confluence_features.py [--dates 20260316,20260317,...] [--workers 14]
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT        = Path("/home/jupiter/Lvl3Quant")
MBO_DIR          = REPO_ROOT / "data/processed/mbo_events_smart_v3"
EXEC_FEAT_DIR    = REPO_ROOT / "output/exec_features_v1"
OUTPUT_DIR       = REPO_ROOT / "output/confluence_features"

# Directories to search for predictions — ordered by preference (newest / best first)
CNN_MAMBA_DIRS = [
    REPO_ROOT / "output/cnn_mamba_v2_smart_v3_mar",
    # Add future CNN-Mamba run dirs here as they are created
]

PATCHTST_DIRS = [
    REPO_ROOT / "output/patchtst_razer_weights",   # covers up to Mar 10
    REPO_ROOT / "output/patchtst_smart_v3_mar",
    # Add future PatchTST run dirs here as they are created
]

# Model parameters (for stride auto-detection fallback)
CNN_MAMBA_WINDOW  = 3000
CNN_MAMBA_STRIDE  = 500   # WINDOW // 6

PATCHTST_WINDOW   = 500
PATCHTST_STRIDE   = 250   # WINDOW // 2

# Execution feature window
EXEC_FEAT_STRIDE  = 5000

# Cost constants (CANONICAL — CLAUDE.md)
ES_RT_COMMISSION_TICKS = 0.376
TICK_VALUE_USD         = 12.50

# MFE/MAE horizon labels available in MBO data
HORIZON_NAMES = ["1s", "5s", "10s", "30s"]

# RTH session for time-of-day bucketing
RTH_OPEN_NS  = 9 * 3600 + 30 * 60    # 09:30 in seconds-from-midnight
RTH_CLOSE_NS = 16 * 3600              # 16:00
TOD_BUCKETS  = 13                     # 30-min buckets: 0=pre, 1..12=RTH, (no close bucket)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("confluence_features")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Discovery — scan dirs and build date→file mapping
# ─────────────────────────────────────────────────────────────────────────────

def _extract_date_from_npz(path: Path) -> Optional[str]:
    """Return YYYYMMDD from oot_files field, or None if unreadable."""
    try:
        d = np.load(path, allow_pickle=True)
        if "oot_files" in d:
            oot = d["oot_files"].tolist()
            if oot:
                fn = str(oot[0]).replace("\\", "/").split("/")[-1]
                date = fn.split("_")[0]
                if len(date) == 8 and date.isdigit():
                    return date
    except Exception:
        pass
    return None


def build_date_to_prediction_map(dirs: List[Path]) -> Dict[str, Path]:
    """
    Scan prediction directories for fold_*_oot_predictions.npz files.
    Returns dict: date_str → Path of NPZ file.
    When multiple dirs have the same date, the FIRST dir in the list wins
    (ordered preference: newest / best model first).
    """
    date_map: Dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for npz in sorted(d.glob("fold_*_oot_predictions.npz")):
            date = _extract_date_from_npz(npz)
            if date and date not in date_map:
                date_map[date] = npz
    return date_map


# ─────────────────────────────────────────────────────────────────────────────
# Stride Auto-Detection (adapted from mfe_mae_path_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_event_indices(
    mbo_path: Path,
    window_size: int,
    default_stride: int,
    target_n_preds: int,
    horizons: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Replay the windowing logic to recover which MBO event index each prediction
    corresponds to (last event of each sliding window).

    Returns:
        event_indices : (N,) int64 — event index of each prediction's label
        timestamps    : (N_events,) int64 nanoseconds
        day_labels    : dict h -> (N_events,) float32 label array
        events        : (N_events, F) float32 raw features
    """
    horizons = horizons or ["1s", "5s", "10s"]
    data = np.load(mbo_path, allow_pickle=True)
    timestamps = data["timestamps"]
    events     = data["events"]
    n_events   = len(timestamps)

    day_labels = {}
    for h in HORIZON_NAMES:
        key = f"labels_{h}"
        if key in data:
            day_labels[h] = data[key].astype(np.float32)

    def _indices_with_stride(s: int) -> np.ndarray:
        starts     = np.arange(0, n_events - window_size + 1, s, dtype=np.int64)
        label_idxs = starts + window_size - 1
        valid      = np.ones(len(label_idxs), dtype=bool)
        for h in horizons:
            if h in day_labels:
                valid &= ~np.isnan(day_labels[h][label_idxs])
        return label_idxs[valid]

    sample_indices = _indices_with_stride(default_stride)

    if abs(len(sample_indices) - target_n_preds) > max(20, 0.01 * target_n_preds):
        # Try candidate strides
        candidates = sorted(set([
            window_size // 2,
            window_size // 4,
            window_size // 6,
            window_size // 8,
            window_size // 10,
            window_size // 12,
            default_stride,
        ]))
        best_stride = default_stride
        best_diff   = abs(len(sample_indices) - target_n_preds)
        for cs in candidates:
            if cs <= 0:
                continue
            test = _indices_with_stride(cs)
            diff = abs(len(test) - target_n_preds)
            if diff < best_diff:
                best_diff   = diff
                best_stride = cs
                sample_indices = test
        if best_stride != default_stride:
            log.debug(f"  stride auto-detected: {best_stride} "
                      f"(got {len(sample_indices)}, target {target_n_preds})")

    return (
        np.array(sample_indices, dtype=np.int64),
        timestamps,
        day_labels,
        events,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp-based Nearest-Neighbor Matching
# ─────────────────────────────────────────────────────────────────────────────

def match_predictions_by_timestamp(
    ts_a: np.ndarray,  # (Na,) ns timestamps for model A
    ts_b: np.ndarray,  # (Nb,) ns timestamps for model B
    max_tol_ns: int = 100_000_000,  # 100ms in nanoseconds
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbor match ts_b to ts_a.

    Returns:
        idx_a   : (M,) indices into ts_a for matched pairs
        idx_b   : (M,) indices into ts_b for matched pairs
        dt_ns   : (M,) time differences (absolute) in nanoseconds
    """
    if len(ts_a) == 0 or len(ts_b) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    # For each element in ts_a, find closest in ts_b via searchsorted
    pos = np.searchsorted(ts_b, ts_a)
    pos = np.clip(pos, 0, len(ts_b) - 1)

    # Also check one index back
    pos_prev = np.maximum(pos - 1, 0)
    dt_cur  = np.abs(ts_b[pos]      - ts_a).astype(np.int64)
    dt_prev = np.abs(ts_b[pos_prev] - ts_a).astype(np.int64)

    better_prev = dt_prev < dt_cur
    final_pos   = np.where(better_prev, pos_prev, pos)
    final_dt    = np.where(better_prev, dt_prev, dt_cur)

    mask = final_dt <= max_tol_ns
    return (
        np.where(mask)[0].astype(np.int64),
        final_pos[mask].astype(np.int64),
        final_dt[mask].astype(np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exec Features Alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_exec_features(
    pred_ts_ns: np.ndarray,   # (N,) prediction timestamps in ns
    exec_npz: Optional[Path], # path to exec_features_v1/{date}_exec_features.npz
) -> Optional[np.ndarray]:
    """
    For each prediction timestamp, look up the exec-feature window that covers it.
    Returns (N, 44) float32 array, or None if exec features unavailable.
    """
    if exec_npz is None or not exec_npz.exists():
        return None

    try:
        d = np.load(exec_npz, allow_pickle=True)
        features  = d["features"].astype(np.float32)   # (W, 44)
        n_windows = features.shape[0]
        stride    = int(d.get("decision_stride", EXEC_FEAT_STRIDE))
    except Exception as e:
        log.warning(f"  Could not load exec features: {e}")
        return None

    # Map each prediction event-index to a window index
    # We reconstruct approximate event indices from timestamps by getting
    # the MBO data — but we already have timestamps for predictions.
    # Instead, use window-index = event_index // stride, clamped.
    # We don't have exact event_indices here, but we can do:
    #   window_index_approx = (rank among predictions / total) * n_windows
    # Better: we pass timestamps and compute ranks
    # Most robust: linearly interpolate window index from timestamp position
    # within the day's timestamp range.
    # But we don't load MBO just for this — use uniform interpolation.

    # Approach: use uniform index (0..N-1) / N * n_windows → window index
    N = len(pred_ts_ns)
    if N == 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32)

    # Use the normalized position within the sorted prediction timestamps
    # (already sorted by construction)
    t_min = pred_ts_ns[0]
    t_max = pred_ts_ns[-1]
    if t_max == t_min:
        widx = np.zeros(N, dtype=np.int64)
    else:
        frac = (pred_ts_ns - t_min).astype(np.float64) / (t_max - t_min)
        widx = np.clip((frac * n_windows).astype(np.int64), 0, n_windows - 1)

    return features[widx]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def day_percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank within the array (0..1). Handles NaN by ignoring them."""
    out = np.full(len(arr), np.nan, dtype=np.float32)
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return out
    vals = arr[valid]
    # rank / (n-1) gives uniform 0..1
    ranks  = np.argsort(np.argsort(vals)).astype(np.float32)
    n      = len(vals)
    if n > 1:
        ranks /= (n - 1)
    out[valid] = ranks
    return out


def compute_rolling_vol(
    timestamps_ns: np.ndarray,
    labels_1s:     np.ndarray,
    pred_indices:  np.ndarray,
    window_s: float = 60.0,
) -> np.ndarray:
    """
    Compute rolling 60-second realized vol at each prediction point.
    Uses the 1s label values as a proxy for returns (tick-denominated moves).
    Returns (N,) float32.
    """
    N      = len(pred_indices)
    out    = np.full(N, np.nan, dtype=np.float32)
    win_ns = int(window_s * 1e9)

    for i, idx in enumerate(pred_indices):
        t_end   = timestamps_ns[idx]
        t_start = t_end - win_ns

        # Vectorized slice: find events in window
        lo = np.searchsorted(timestamps_ns, t_start, side="left")
        hi = idx + 1  # inclusive of current event

        window_labels = labels_1s[lo:hi]
        valid = window_labels[~np.isnan(window_labels)]
        if len(valid) > 2:
            out[i] = float(np.std(valid))

    return out


def compute_book_imbalance(
    events: np.ndarray,       # (N_events, F) raw MBO features
    pred_indices: np.ndarray, # (M,) event indices
) -> np.ndarray:
    """
    Rough bid/ask imbalance from MBO event features.
    Feature layout from train_cnn_mamba.py docstring:
      col 0: time_delta_log
      col 1: event_type_id   (0=trade, 1=add, 2=cancel, 3=modify, 4=clear)
      col 2: side_id          (0=bid, 1=ask, -1=neutral)
      col 3: price_rel_ticks
      col 4: qty_log
      col 5: spread_ticks

    We compute a rolling count imbalance: fraction of recent add-events on bid side.
    """
    N   = len(pred_indices)
    out = np.full(N, np.nan, dtype=np.float32)

    if events.shape[1] < 5:
        return out

    event_type = events[:, 1]  # ADD_ORDER = 1
    side       = events[:, 2]  # bid=0, ask=1, neutral=-1

    LOOKBACK = 500  # events to look back

    for i, idx in enumerate(pred_indices):
        lo  = max(0, idx - LOOKBACK)
        seg = events[lo:idx + 1]
        add_mask  = seg[:, 1] == 1.0  # ADD events
        adds      = seg[add_mask]
        if len(adds) == 0:
            out[i] = 0.0
            continue
        bid_adds  = (adds[:, 2] == 0.0).sum()  # side==0 → bid
        ask_adds  = (adds[:, 2] == 1.0).sum()  # side==1 → ask
        total     = bid_adds + ask_adds
        if total > 0:
            out[i] = float(bid_adds - ask_adds) / float(total)

    return out


def compute_tod_bucket(timestamps_ns: np.ndarray, pred_indices: np.ndarray) -> np.ndarray:
    """
    Time-of-day bucket: 0=pre-RTH, 1..12=30-min buckets 09:30-16:00.
    Bucket k=1 → 09:30-10:00, k=12 → 15:30-16:00.
    """
    N   = len(pred_indices)
    out = np.zeros(N, dtype=np.int8)
    for i, idx in enumerate(pred_indices):
        ts_s = (timestamps_ns[idx] // 1_000_000_000) % 86400  # seconds from midnight UTC-5
        # Approximate ET: subtract 5h = 18000s (valid during EST; EDT=14400s)
        # Use 18000 as conservative constant (EST)
        ts_s_et = ts_s  # timestamps already appear to be in local time based on prior checks
        secs_from_open = ts_s_et - RTH_OPEN_NS
        if secs_from_open < 0:
            out[i] = 0
        else:
            bucket = int(secs_from_open // 1800) + 1  # 1-based
            out[i] = min(bucket, TOD_BUCKETS)
    return out


def compute_mfe_mae_from_path(
    pred_values: np.ndarray,         # (N,) 1s predictions (for direction)
    day_labels:  Dict[str, np.ndarray],
    pred_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute MFE/MAE at each horizon using the available label checkpoints.
    Returns dict with keys mfe_1s, mae_1s, mfe_5s, mae_5s, mfe_10s, mae_10s.
    All in ticks (same units as labels).
    """
    N       = len(pred_indices)
    results = {
        "mfe_1s":  np.full(N, np.nan, np.float32),
        "mae_1s":  np.full(N, np.nan, np.float32),
        "mfe_5s":  np.full(N, np.nan, np.float32),
        "mae_5s":  np.full(N, np.nan, np.float32),
        "mfe_10s": np.full(N, np.nan, np.float32),
        "mae_10s": np.full(N, np.nan, np.float32),
    }

    # Available horizons from MBO data
    avail = {}
    for h in ["1s", "5s", "10s", "30s"]:
        if h in day_labels:
            avail[h] = day_labels[h]

    if not avail:
        return results

    direction = np.sign(pred_values)
    direction[direction == 0] = 1.0

    horizon_order = ["1s", "5s", "10s", "30s"]
    checkpoints   = [h for h in horizon_order if h in avail]

    for i, idx in enumerate(pred_indices):
        d     = direction[i]
        path  = np.array([avail[h][idx] for h in checkpoints])
        valid = ~np.isnan(path)

        if valid.sum() < 2:
            continue

        dir_path = path * d  # positive = favorable

        # MFE at each horizon = cumulative max up to that checkpoint
        for j, h in enumerate(checkpoints):
            if not valid[j]:
                continue
            seg         = dir_path[: j + 1][~np.isnan(path[: j + 1])]
            mfe_h       = float(max(0.0, np.max(seg)))
            mae_h       = float(abs(min(0.0, np.min(seg))))
            key_mfe     = f"mfe_{h}"
            key_mae     = f"mae_{h}"
            if key_mfe in results:
                results[key_mfe][i] = mfe_h
                results[key_mae][i] = mae_h

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-Date Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_date(args: Tuple) -> Optional[Dict]:
    """
    Worker function (runs in subprocess via Pool).
    Returns feature dict for the date or None on failure.
    """
    (
        date,
        cnn_path,
        ptst_path,
        mbo_path,
        exec_feat_path,
        output_path,
    ) = args

    t0 = time.time()
    pid = os.getpid()

    # ── Load predictions ──────────────────────────────────────────────────
    try:
        cnn_npz  = np.load(cnn_path,  allow_pickle=True) if cnn_path  else None
        ptst_npz = np.load(ptst_path, allow_pickle=True) if ptst_path else None
    except Exception as e:
        logging.getLogger("confluence_features").warning(
            f"[{pid}] {date} failed to load predictions: {e}")
        return None

    if cnn_npz is None and ptst_npz is None:
        return None

    # ── Load MBO data ─────────────────────────────────────────────────────
    try:
        mbo_data = np.load(mbo_path, allow_pickle=True)
    except Exception as e:
        logging.getLogger("confluence_features").warning(
            f"[{pid}] {date} failed to load MBO data: {e}")
        return None

    # ── Reconstruct event indices for CNN-Mamba ───────────────────────────
    cnn_preds    = None
    cnn_labels   = None
    cnn_ts_ns    = None
    cnn_indices  = None
    day_labels   = None

    if cnn_npz is not None:
        n_cnn = cnn_npz["predictions"].shape[0]
        try:
            cnn_indices, all_timestamps, day_labels, raw_events = reconstruct_event_indices(
                mbo_path,
                window_size   = CNN_MAMBA_WINDOW,
                default_stride= CNN_MAMBA_STRIDE,
                target_n_preds= n_cnn,
                horizons      = ["1s", "5s", "10s"],
            )
        except Exception as e:
            logging.getLogger("confluence_features").warning(
                f"[{pid}] {date} CNN stride reconstruction failed: {e}")
            return None

        if abs(len(cnn_indices) - n_cnn) > max(50, 0.02 * n_cnn):
            logging.getLogger("confluence_features").warning(
                f"[{pid}] {date} CNN index count mismatch: "
                f"got {len(cnn_indices)}, expected {n_cnn} — skipping")
            return None

        n_use    = min(len(cnn_indices), n_cnn)
        cnn_indices  = cnn_indices[:n_use]
        cnn_preds    = cnn_npz["predictions"][:n_use]       # (N, 3) horizons: 1s,5s,10s
        cnn_labels   = cnn_npz["labels"][:n_use]            # (N, 3) true moves
        cnn_ts_ns    = all_timestamps[cnn_indices]          # (N,) nanoseconds

    # ── Reconstruct event indices for PatchTST ────────────────────────────
    ptst_preds   = None
    ptst_ts_ns   = None
    ptst_indices = None

    if ptst_npz is not None:
        n_ptst = ptst_npz["predictions"].shape[0]
        try:
            if day_labels is None:
                ptst_indices, all_timestamps, day_labels, raw_events = reconstruct_event_indices(
                    mbo_path,
                    window_size   = PATCHTST_WINDOW,
                    default_stride= PATCHTST_STRIDE,
                    target_n_preds= n_ptst,
                    horizons      = ["1s", "5s", "10s"],
                )
            else:
                ptst_indices, _ts, _, _ = reconstruct_event_indices(
                    mbo_path,
                    window_size   = PATCHTST_WINDOW,
                    default_stride= PATCHTST_STRIDE,
                    target_n_preds= n_ptst,
                    horizons      = ["1s", "5s", "10s"],
                )
        except Exception as e:
            logging.getLogger("confluence_features").warning(
                f"[{pid}] {date} PatchTST stride reconstruction failed: {e}")
            ptst_npz = None

        if ptst_npz is not None:
            if abs(len(ptst_indices) - n_ptst) > max(50, 0.02 * n_ptst):
                logging.getLogger("confluence_features").warning(
                    f"[{pid}] {date} PatchTST index count mismatch: "
                    f"got {len(ptst_indices)}, expected {n_ptst} — ignoring PatchTST")
                ptst_npz = None
            else:
                n_use     = min(len(ptst_indices), n_ptst)
                ptst_indices = ptst_indices[:n_use]
                ptst_preds   = ptst_npz["predictions"][:n_use]  # (N, 3)
                ptst_ts_ns   = all_timestamps[ptst_indices]      # (N,)

    # ── Need at least one model ───────────────────────────────────────────
    if cnn_preds is None and ptst_preds is None:
        return None

    # ── Determine anchor arrays (CNN primary, PatchTST secondary) ─────────
    # We build output rows from CNN positions when available, else PatchTST only.
    # When both exist, we match PatchTST to CNN timestamps.

    if cnn_preds is not None:
        N          = len(cnn_preds)
        anchor_ts  = cnn_ts_ns
        anchor_idx = cnn_indices
        base_preds = cnn_preds   # (N, 3): cols = 1s, 5s, 10s
        base_lbls  = cnn_labels
        base_day_labels = day_labels
    else:
        N          = len(ptst_preds)
        anchor_ts  = ptst_ts_ns
        anchor_idx = ptst_indices
        base_preds = np.full((N, 3), np.nan, np.float32)
        base_lbls  = ptst_npz["labels"][:N]
        base_day_labels = day_labels

    # ── Match PatchTST to anchor ──────────────────────────────────────────
    ptst_at_anchor = np.full((N, 3), np.nan, np.float32)

    if ptst_preds is not None and anchor_ts is not None and ptst_ts_ns is not None:
        ia, ib, dt = match_predictions_by_timestamp(anchor_ts, ptst_ts_ns)
        if len(ia) > 0:
            ptst_at_anchor[ia] = ptst_preds[ib]

    # ── CNN predictions at each horizon ───────────────────────────────────
    cnn_pred_1s  = base_preds[:, 0] if base_preds is not None else np.full(N, np.nan, np.float32)
    cnn_pred_5s  = base_preds[:, 1] if base_preds is not None else np.full(N, np.nan, np.float32)
    cnn_pred_10s = base_preds[:, 2] if base_preds is not None else np.full(N, np.nan, np.float32)

    ptst_pred_1s  = ptst_at_anchor[:, 0]
    ptst_pred_5s  = ptst_at_anchor[:, 1]
    ptst_pred_10s = ptst_at_anchor[:, 2]

    # ── Actual moves (from labels) ─────────────────────────────────────────
    actual_move_1s  = base_lbls[:, 0] if base_lbls is not None else np.full(N, np.nan, np.float32)
    actual_move_5s  = base_lbls[:, 1] if base_lbls is not None else np.full(N, np.nan, np.float32)
    actual_move_10s = base_lbls[:, 2] if base_lbls is not None else np.full(N, np.nan, np.float32)

    # ── Confluence features ───────────────────────────────────────────────
    # Agreement on direction at 1s horizon (primary)
    cnn_sign  = np.sign(cnn_pred_1s)
    ptst_sign = np.sign(ptst_pred_1s)
    both_valid = (~np.isnan(cnn_pred_1s)) & (~np.isnan(ptst_pred_1s))

    confluence_agree = np.zeros(N, dtype=np.float32)
    confluence_agree[both_valid & (cnn_sign == ptst_sign)] = 1.0
    confluence_agree[~both_valid] = np.nan

    # Confluence strength: product of magnitudes when agreeing, negative when opposing
    confluence_strength = np.full(N, np.nan, dtype=np.float32)
    agree_mask  = both_valid & (cnn_sign == ptst_sign) & (cnn_sign != 0)
    oppose_mask = both_valid & (cnn_sign != ptst_sign)
    confluence_strength[agree_mask]  =  np.abs(cnn_pred_1s[agree_mask]) * np.abs(ptst_pred_1s[agree_mask])
    confluence_strength[oppose_mask] = -np.abs(cnn_pred_1s[oppose_mask]) * np.abs(ptst_pred_1s[oppose_mask])

    # Confidence percentile ranks (within day)
    cnn_confidence_pct  = day_percentile_rank(np.abs(cnn_pred_1s))
    ptst_confidence_pct = day_percentile_rank(np.abs(ptst_pred_1s))

    # Combined confidence: geometric mean of percentile ranks
    combined_confidence = np.full(N, np.nan, dtype=np.float32)
    both_conf = (~np.isnan(cnn_confidence_pct)) & (~np.isnan(ptst_confidence_pct))
    combined_confidence[both_conf] = np.sqrt(
        cnn_confidence_pct[both_conf] * ptst_confidence_pct[both_conf]
    )
    # Fall back to single model when only one available
    only_cnn  = (~np.isnan(cnn_confidence_pct)) & np.isnan(ptst_confidence_pct)
    only_ptst = np.isnan(cnn_confidence_pct) & (~np.isnan(ptst_confidence_pct))
    combined_confidence[only_cnn]  = cnn_confidence_pct[only_cnn]
    combined_confidence[only_ptst] = ptst_confidence_pct[only_ptst]

    # ── Rolling volatility (60s) ──────────────────────────────────────────
    if base_day_labels and "1s" in base_day_labels:
        vol_state = compute_rolling_vol(
            all_timestamps, base_day_labels["1s"], anchor_idx, window_s=60.0
        )
    else:
        vol_state = np.full(N, np.nan, dtype=np.float32)

    # ── Book imbalance ────────────────────────────────────────────────────
    book_imbalance = compute_book_imbalance(raw_events, anchor_idx)

    # ── Time-of-day bucket ────────────────────────────────────────────────
    tod_bucket = compute_tod_bucket(all_timestamps, anchor_idx)

    # ── MFE / MAE ─────────────────────────────────────────────────────────
    mfe_mae = compute_mfe_mae_from_path(
        cnn_pred_1s, base_day_labels or {}, anchor_idx
    )

    # ── Exec features (market microstructure) ─────────────────────────────
    exec_feats  = align_exec_features(anchor_ts, exec_feat_path)  # (N, 44) or None

    # ── Assemble output dict ──────────────────────────────────────────────
    result = {
        "date"               : np.array(date),
        "timestamps_ns"      : anchor_ts.astype(np.int64),
        # Raw predictions
        "cnn_pred_1s"        : cnn_pred_1s.astype(np.float32),
        "cnn_pred_5s"        : cnn_pred_5s.astype(np.float32),
        "cnn_pred_10s"       : cnn_pred_10s.astype(np.float32),
        "ptst_pred_1s"       : ptst_pred_1s.astype(np.float32),
        "ptst_pred_5s"       : ptst_pred_5s.astype(np.float32),
        "ptst_pred_10s"      : ptst_pred_10s.astype(np.float32),
        # Confluence
        "confluence_agree"   : confluence_agree.astype(np.float32),
        "confluence_strength": confluence_strength.astype(np.float32),
        "cnn_confidence_pct" : cnn_confidence_pct.astype(np.float32),
        "ptst_confidence_pct": ptst_confidence_pct.astype(np.float32),
        "combined_confidence": combined_confidence.astype(np.float32),
        # Market state
        "vol_state"          : vol_state.astype(np.float32),
        "book_imbalance"     : book_imbalance.astype(np.float32),
        "tod_bucket"         : tod_bucket.astype(np.int8),
        # Targets
        "actual_move_1s"     : actual_move_1s.astype(np.float32),
        "actual_move_5s"     : actual_move_5s.astype(np.float32),
        "actual_move_10s"    : actual_move_10s.astype(np.float32),
        # MFE / MAE
        "mfe_1s"             : mfe_mae["mfe_1s"],
        "mae_1s"             : mfe_mae["mae_1s"],
        "mfe_5s"             : mfe_mae["mfe_5s"],
        "mae_5s"             : mfe_mae["mae_5s"],
        "mfe_10s"            : mfe_mae["mfe_10s"],
        "mae_10s"            : mfe_mae["mae_10s"],
    }

    if exec_feats is not None:
        result["exec_features"] = exec_feats.astype(np.float32)

    # ── Save per-date ─────────────────────────────────────────────────────
    try:
        np.savez_compressed(output_path, **result)
    except Exception as e:
        logging.getLogger("confluence_features").error(
            f"[{pid}] {date} failed to save: {e}")
        return None

    elapsed = time.time() - t0
    n_agree  = int(np.nansum(confluence_agree))
    n_valid  = int(np.sum(~np.isnan(confluence_agree)))
    rate     = n_agree / n_valid if n_valid > 0 else 0.0

    logging.getLogger("confluence_features").info(
        f"  {date}: N={N}, confluence={rate:.2%}, "
        f"both_models={int(both_valid.sum())}/{N}, "
        f"exec_feats={'yes' if exec_feats is not None else 'no'}, "
        f"elapsed={elapsed:.1f}s"
    )

    return {
        "date"          : date,
        "n_samples"     : N,
        "n_agree"       : n_agree,
        "n_valid_agree" : n_valid,
        "both_models"   : int(both_valid.sum()),
        "has_exec_feats": exec_feats is not None,
        "output_path"   : str(output_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build confluence features for execution models")
    parser.add_argument("--dates",   type=str, default=None,
                        help="Comma-separated YYYYMMDD dates to process (default: all available)")
    parser.add_argument("--workers", type=int, default=14,
                        help="Number of parallel workers (default: 14)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess dates that already have output files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build date maps ───────────────────────────────────────────────────
    log.info("Scanning prediction directories...")
    cnn_map  = build_date_to_prediction_map(CNN_MAMBA_DIRS)
    ptst_map = build_date_to_prediction_map(PATCHTST_DIRS)

    log.info(f"  CNN-Mamba dates: {len(cnn_map)}: {sorted(cnn_map.keys())}")
    log.info(f"  PatchTST dates:  {len(ptst_map)}: {sorted(ptst_map.keys())}")

    # Union of all dates with at least one model
    all_pred_dates = sorted(set(cnn_map) | set(ptst_map))

    # MBO dates available
    mbo_dates = {
        f.name.split("_")[0]
        for f in MBO_DIR.glob("*_mbo_events.npz")
    }

    # Exec feature dates
    exec_dates = {
        f.name.split("_")[0]
        for f in EXEC_FEAT_DIR.glob("*_exec_features.npz")
    }

    # Filter: must have MBO data
    valid_dates = sorted(d for d in all_pred_dates if d in mbo_dates)

    if args.dates:
        requested = set(args.dates.split(","))
        valid_dates = sorted(d for d in valid_dates if d in requested)

    if not valid_dates:
        log.warning("No processable dates found. Check prediction and MBO directories.")
        log.info(f"  Prediction dates (CNN+PTST): {all_pred_dates}")
        log.info(f"  MBO dates available: {sorted(mbo_dates)[:10]}...")
        sys.exit(0)

    log.info(f"\nDates to process: {len(valid_dates)}")
    for d in valid_dates:
        has_cnn  = "CNN" if d in cnn_map  else "   "
        has_ptst = "PTST" if d in ptst_map else "    "
        has_exec = "exec" if d in exec_dates else "    "
        log.info(f"  {d}  [{has_cnn}] [{has_ptst}] [{has_exec}]")

    # ── Build work list ───────────────────────────────────────────────────
    work = []
    for date in valid_dates:
        out_path = OUTPUT_DIR / f"{date}_confluence.npz"
        if out_path.exists() and not args.overwrite:
            log.info(f"  Skipping {date} (already exists, use --overwrite)")
            continue
        work.append((
            date,
            cnn_map.get(date),
            ptst_map.get(date),
            MBO_DIR / f"{date}_mbo_events.npz",
            EXEC_FEAT_DIR / f"{date}_exec_features.npz" if date in exec_dates else None,
            out_path,
        ))

    if not work:
        log.info("All dates already processed.")
    else:
        log.info(f"\nProcessing {len(work)} dates with {args.workers} workers...")
        t_start = time.time()

        results = []
        if args.workers > 1 and len(work) > 1:
            with Pool(processes=args.workers) as pool:
                for r in pool.imap_unordered(process_date, work):
                    results.append(r)
        else:
            for w in work:
                results.append(process_date(w))

        elapsed = time.time() - t_start
        ok      = [r for r in results if r is not None]
        log.info(f"\nProcessed {len(ok)}/{len(work)} dates in {elapsed:.1f}s")

    # ── Combine all per-date files ─────────────────────────────────────────
    log.info("\nCombining all per-date files...")
    combined = {}
    date_files = sorted(OUTPUT_DIR.glob("*_confluence.npz"))

    all_keys = None
    per_date_arrays = []

    for f in date_files:
        try:
            d = np.load(f, allow_pickle=True)
            if all_keys is None:
                all_keys = list(d.keys())
            per_date_arrays.append({k: d[k] for k in all_keys if k in d})
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    if per_date_arrays and all_keys:
        for k in all_keys:
            if k == "date":
                continue
            arrays = [row[k] for row in per_date_arrays if k in row and row[k].ndim > 0]
            if arrays:
                try:
                    combined[k] = np.concatenate(arrays, axis=0)
                except Exception as e:
                    log.warning(f"  Could not concatenate {k}: {e}")

        # Add per-date labels for slicing
        dates_arr = np.concatenate([
            np.full(len(row.get("timestamps_ns", np.array([]))), row["date"])
            for row in per_date_arrays if "timestamps_ns" in row
        ])
        combined["date"] = dates_arr

        combined_path = OUTPUT_DIR / "all_confluence_features.npz"
        np.savez_compressed(combined_path, **combined)
        log.info(f"Combined file: {combined_path}")
        total = len(combined.get("timestamps_ns", []))
        log.info(f"  Total samples: {total:,}")

    # ── Summary stats ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY STATISTICS")
    log.info("=" * 60)

    if combined:
        total = len(combined.get("timestamps_ns", np.array([])))
        log.info(f"Total samples:        {total:,}")

        agree = combined.get("confluence_agree")
        if agree is not None:
            valid_mask = ~np.isnan(agree)
            agree_rate = float(np.nanmean(agree))
            log.info(f"Confluence agreement: {agree_rate:.2%} "
                     f"(n={int(valid_mask.sum()):,})")

        conf = combined.get("combined_confidence")
        if conf is not None and agree is not None:
            for label, lo, hi in [
                ("Top 10% conf",  0.9, 1.0),
                ("Top 20% conf",  0.8, 1.0),
                ("Mid conf",      0.4, 0.6),
                ("Low conf",      0.0, 0.2),
            ]:
                mask = (~np.isnan(conf)) & (conf >= lo) & (conf < hi)
                if mask.sum() > 0:
                    a = float(np.nanmean(agree[mask]))
                    log.info(f"  {label:15s}: n={mask.sum():6,}  agreement={a:.2%}")

        cnn1 = combined.get("cnn_pred_1s")
        if cnn1 is not None:
            valid = ~np.isnan(cnn1)
            log.info(f"\nCNN-Mamba 1s pred:  n={valid.sum():,}, "
                     f"mean={np.nanmean(cnn1):.4f}, std={np.nanstd(cnn1):.4f}")

        ptst1 = combined.get("ptst_pred_1s")
        if ptst1 is not None:
            valid = ~np.isnan(ptst1)
            log.info(f"PatchTST 1s pred:   n={valid.sum():,}, "
                     f"mean={np.nanmean(ptst1):.4f}, std={np.nanstd(ptst1):.4f}")

        mfe1 = combined.get("mfe_1s")
        if mfe1 is not None:
            log.info(f"\nMFE@1s:  mean={np.nanmean(mfe1):.4f}t, "
                     f"p90={np.nanpercentile(mfe1, 90):.4f}t")
        mae1 = combined.get("mae_1s")
        if mae1 is not None:
            log.info(f"MAE@1s:  mean={np.nanmean(mae1):.4f}t, "
                     f"p90={np.nanpercentile(mae1, 90):.4f}t")

    log.info("=" * 60)
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("Done.")


if __name__ == "__main__":
    main()
