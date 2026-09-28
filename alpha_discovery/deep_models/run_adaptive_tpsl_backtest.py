#!/usr/bin/env python3
"""
Adaptive TP/SL Backtest — Tests hypothesis that TP/SL must adapt to
time-of-day, volatility, and volume.

Hypothesis: "Getting a tick move at 2am ET is much slower than at 2pm ET.
Volume and volatility determine speed of moves. Having a static set on a
low vol day would be slower and not hit profit as fast."

Configs:
  A: Time-of-Day Adaptive TP/SL
  B: Volatility-Adaptive TP/SL
  C: Combined ToD + Vol
  D: PatchTST Confluence (CNN Mamba v2 + PatchTST agree, PatchTST reversal exit)
  E: Static Baseline (CNN Mamba v2 only)

CPU-only (Jupiter, 64GB RAM). Walk-forward OOT predictions only.
"""

import json
import os
import sys
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
CNN_DIR = BASE / "output" / "cnn_mamba_v2_smart_v3_mar"
PTST_DIR = BASE / "output" / "patchtst_smart_v3_mar"
VOL_DIR = BASE / "output" / "vol_lgbm_v3"
SRC_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"

# ── Constants ────────────────────────────────────────────────────────
TICK_VAL = 12.50          # USD per tick (NQ)
COMMISSION = 0.376        # ticks, one-way
SPREAD = 0.5              # ticks, one-way
ROUND_TRIP_COST = 2 * (COMMISSION + SPREAD)  # 1.752 ticks total

# Model stride/anchor configs (verified empirically)
CNN_STRIDE = 500
CNN_FIRST_ANCHOR = 999
PTST_STRIDE = 250
PTST_FIRST_ANCHOR = 499

HORIZON_NAMES = ["1s", "5s", "10s"]
HORIZON_IDX = {h: i for i, h in enumerate(HORIZON_NAMES)}

# Z-score rolling window
ZSCORE_WINDOW = 3000

# ── Confidence Tiers ─────────────────────────────────────────────────
CONFIDENCE_TIERS = {
    "All":     0.0,
    "Top50%":  0.50,
    "Top25%":  0.75,
    "Top10%":  0.90,
    "Top5%":   0.95,
    "Top1%":   0.99,
}

# ── Session Definitions (ET = UTC-4 for EDT, UTC-5 for EST) ─────────
# Using UTC offsets; trading dates in Feb/Mar 2026 are EST (UTC-5)
# CME NQ futures: Sun 6pm ET - Fri 5pm ET
# Pre-market:  2am - 9:30am ET  -> 7:00 - 14:30 UTC
# RTH Open:    9:30 - 10:30am   -> 14:30 - 15:30 UTC
# RTH Core:    10:30am - 3pm    -> 15:30 - 20:00 UTC
# RTH Close:   3pm - 4pm        -> 20:00 - 21:00 UTC
# Post-market: 4pm+             -> 21:00+ UTC
# Note: Dates in our data (Feb-Mar 2026) are EST (UTC-5)
ET_OFFSET_HOURS = -5  # EST

SESSION_BOUNDS_ET = [
    ("pre_market",   2.0,  9.5),   # 2:00am - 9:30am ET
    ("rth_open",     9.5,  10.5),  # 9:30am - 10:30am ET
    ("rth_core",     10.5, 15.0),  # 10:30am - 3:00pm ET
    ("rth_close",    15.0, 16.0),  # 3:00pm - 4:00pm ET
    ("post_market",  16.0, 24.0),  # 4:00pm+ ET
]

# ── Config A: Time-of-Day Adaptive TP/SL ────────────────────────────
TOD_TPSL = {
    "pre_market":  {"tp": 12, "sl": 8, "timeout_s": 60},
    "rth_open":    {"tp": 6,  "sl": 4, "timeout_s": 15},
    "rth_core":    {"tp": 8,  "sl": 5, "timeout_s": 30},
    "rth_close":   {"tp": 6,  "sl": 4, "timeout_s": 15},
    "post_market": {"tp": 12, "sl": 8, "timeout_s": 60},
}

# ── Config B: Volatility-Adaptive TP/SL ─────────────────────────────
VOL_TPSL = {
    "low":    {"tp": 12, "sl": 8, "timeout_s": 60},
    "medium": {"tp": 8,  "sl": 5, "timeout_s": 30},
    "high":   {"tp": 5,  "sl": 3, "timeout_s": 10},
}

# ── Config C: Combined multipliers ──────────────────────────────────
VOL_MULTIPLIER = {"low": 1.5, "medium": 1.0, "high": 0.6}

# ── Config E: Static Baseline ───────────────────────────────────────
STATIC_TP = 8
STATIC_SL = 5
STATIC_TIMEOUT_S = 30


# ── Data Structures ──────────────────────────────────────────────────
@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int           # +1 long, -1 short
    entry_ts_ns: int
    exit_ts_ns: int
    hold_time_s: float
    pnl_ticks_gross: float
    pnl_ticks_net: float     # after costs
    exit_reason: str         # "tp", "sl", "timeout", "ptst_reversal"
    session: str
    mfe_ticks: float         # max favorable excursion
    mae_ticks: float         # max adverse excursion
    confidence_z: float      # abs z-score at entry
    fold: int
    date: str
    horizon: str


@dataclass
class ConfigResult:
    config_name: str
    horizon: str
    tier: str
    total_pnl_ticks: float = 0.0
    total_pnl_usd: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    sortino: float = 0.0
    avg_hold_s: float = 0.0
    tp_rate: float = 0.0
    sl_rate: float = 0.0
    timeout_rate: float = 0.0
    ptst_reversal_rate: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    per_session: Dict[str, Dict] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────

def ts_ns_to_et_hour(ts_ns: int) -> float:
    """Convert nanosecond timestamp to fractional hour in ET."""
    utc_sec = ts_ns / 1e9
    et_sec = utc_sec + ET_OFFSET_HOURS * 3600
    # Get hour of day
    dt = datetime.fromtimestamp(et_sec, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def get_session(et_hour: float) -> str:
    """Classify fractional ET hour into session."""
    for name, start, end in SESSION_BOUNDS_ET:
        if start <= et_hour < end:
            return name
    # Before 2am ET or edge cases
    if et_hour < 2.0:
        return "post_market"  # overnight, treat as thin
    return "post_market"


def classify_vol(vol_pred: float, vol_p25: float, vol_p75: float) -> str:
    """Classify volatility prediction into low/medium/high."""
    if vol_pred < vol_p25:
        return "low"
    elif vol_pred > vol_p75:
        return "high"
    else:
        return "medium"


def compute_zscore_rolling(preds: np.ndarray, window: int = ZSCORE_WINDOW) -> np.ndarray:
    """Compute rolling z-score for each prediction."""
    n = len(preds)
    zscores = np.zeros(n)
    # Use cumulative stats for efficiency
    cumsum = np.cumsum(preds)
    cumsum2 = np.cumsum(preds ** 2)

    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 30:  # need minimum samples
            zscores[i] = 0.0
            continue
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
        s2 = cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        zscores[i] = (preds[i] - mean) / std

    return zscores


def compute_sortino(returns: np.ndarray, target: float = 0.0) -> float:
    """Compute Sortino ratio from array of per-trade returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - target
    mean_excess = np.mean(excess)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if mean_excess > 0 else 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return float("inf") if mean_excess > 0 else 0.0
    return mean_excess / downside_std


def discover_fold_date_mapping() -> Dict[str, Dict[str, Any]]:
    """
    Discover which dates have both CNN Mamba v2 and PatchTST predictions.
    Returns dict keyed by date string with fold indices and paths.
    """
    # CNN Mamba v2 fold -> date
    cnn_folds = {}
    for p in sorted(CNN_DIR.glob("fold_*_oot_predictions.npz")):
        fold_idx = int(p.name.split("_")[1])
        d = np.load(p, allow_pickle=True)
        oot_file = str(d["oot_files"][0])
        date_match = re.search(r"(\d{8})", os.path.basename(oot_file))
        if date_match:
            cnn_folds[date_match.group(1)] = {"fold": fold_idx, "path": p}

    # PatchTST fold -> date
    ptst_folds = {}
    for p in sorted(PTST_DIR.glob("fold_*_oot_predictions.npz")):
        fold_idx = int(p.name.split("_")[1])
        d = np.load(p, allow_pickle=True)
        oot_file = str(d["oot_files"][0])
        date_match = re.search(r"(\d{8})", os.path.basename(oot_file))
        if date_match:
            ptst_folds[date_match.group(1)] = {"fold": fold_idx, "path": p}

    # Vol LGBM -> date
    vol_files = {}
    for p in sorted(VOL_DIR.glob("vol_v3_*_predictions.npz")):
        date_match = re.search(r"(\d{8})", p.name)
        if date_match:
            vol_files[date_match.group(1)] = p

    # Find common dates
    common_dates = set(cnn_folds.keys()) & set(ptst_folds.keys())
    result = {}
    for dt in sorted(common_dates):
        result[dt] = {
            "cnn_fold": cnn_folds[dt]["fold"],
            "cnn_path": cnn_folds[dt]["path"],
            "ptst_fold": ptst_folds[dt]["fold"],
            "ptst_path": ptst_folds[dt]["path"],
            "vol_path": vol_files.get(dt),
            "src_path": SRC_DIR / f"{dt}_mbo_events.npz",
        }
    return result


def load_day_data(date_info: Dict) -> Optional[Dict]:
    """
    Load aligned CNN Mamba v2, PatchTST, vol predictions, and source timestamps
    for a single day. Align predictions to common anchor indices.
    """
    src_path = date_info["src_path"]
    if not src_path.exists():
        print(f"  [SKIP] Source file not found: {src_path}")
        return None

    # Load source timestamps
    src = np.load(src_path, allow_pickle=True)
    timestamps = src["timestamps"]
    labels_1s = src["labels_1s"]
    labels_5s = src["labels_5s"]
    labels_10s = src["labels_10s"]
    n_events = len(timestamps)

    # Load CNN Mamba v2 predictions
    cnn = np.load(date_info["cnn_path"], allow_pickle=True)
    cnn_preds = cnn["predictions"]  # (N_cnn, 3) [1s, 5s, 10s]
    cnn_labels = cnn["labels"]
    n_cnn = len(cnn_preds)

    # CNN anchors: start at CNN_FIRST_ANCHOR, stride CNN_STRIDE
    cnn_anchors = np.arange(CNN_FIRST_ANCHOR, n_events, CNN_STRIDE)[:n_cnn]
    if len(cnn_anchors) != n_cnn:
        print(f"  [WARN] CNN anchor count mismatch: {len(cnn_anchors)} vs {n_cnn}")
        min_n = min(len(cnn_anchors), n_cnn)
        cnn_anchors = cnn_anchors[:min_n]
        cnn_preds = cnn_preds[:min_n]
        cnn_labels = cnn_labels[:min_n]

    # Load PatchTST predictions
    ptst = np.load(date_info["ptst_path"], allow_pickle=True)
    ptst_preds = ptst["predictions"]  # (N_ptst, 3)
    n_ptst = len(ptst_preds)

    # PatchTST anchors: start at PTST_FIRST_ANCHOR, stride PTST_STRIDE
    ptst_anchors = np.arange(PTST_FIRST_ANCHOR, n_events, PTST_STRIDE)[:n_ptst]
    if len(ptst_anchors) != n_ptst:
        print(f"  [WARN] PatchTST anchor count mismatch: {len(ptst_anchors)} vs {n_ptst}")
        min_n = min(len(ptst_anchors), n_ptst)
        ptst_anchors = ptst_anchors[:min_n]
        ptst_preds = ptst_preds[:min_n]

    # Load vol predictions (optional)
    vol_preds = None
    vol_anchors = None
    if date_info["vol_path"] and date_info["vol_path"].exists():
        vol = np.load(date_info["vol_path"], allow_pickle=True)
        vol_preds = vol["predictions"]  # (N_vol, 3) [10s, 30s, 60s horizons]
        vol_anchors_raw = vol["anchor_idxs"]
        # Vol LGBM has its own anchor_idxs
        vol_anchors = vol_anchors_raw[:len(vol_preds)]

    # Align: use CNN anchors as primary grid, map PatchTST to nearest CNN anchor
    # Build PatchTST lookup: for each CNN anchor, find the nearest PatchTST prediction
    ptst_at_cnn = np.full((n_cnn, 3), np.nan, dtype=np.float32)
    if len(ptst_anchors) > 0:
        # For each CNN anchor, find nearest PatchTST anchor via searchsorted
        insert_pos = np.searchsorted(ptst_anchors, cnn_anchors)
        for i, pos in enumerate(insert_pos):
            # Check left and right neighbors
            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < len(ptst_anchors):
                candidates.append(pos)
            if not candidates:
                continue
            best = min(candidates, key=lambda c: abs(ptst_anchors[c] - cnn_anchors[i]))
            # Only use if within 1 CNN stride distance
            if abs(ptst_anchors[best] - cnn_anchors[i]) <= CNN_STRIDE:
                ptst_at_cnn[i] = ptst_preds[best]

    # Vol lookup: for each CNN anchor, find nearest vol prediction
    vol_at_cnn = np.full(n_cnn, np.nan, dtype=np.float32)
    if vol_preds is not None and vol_anchors is not None and len(vol_anchors) > 0:
        insert_pos = np.searchsorted(vol_anchors, cnn_anchors)
        for i, pos in enumerate(insert_pos):
            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < len(vol_anchors):
                candidates.append(pos)
            if not candidates:
                continue
            best = min(candidates, key=lambda c: abs(vol_anchors[c] - cnn_anchors[i]))
            if abs(vol_anchors[best] - cnn_anchors[i]) <= CNN_STRIDE:
                vol_at_cnn[i] = vol_preds[best, 0]  # Use 10s vol horizon

    # Timestamps at CNN anchors
    ts_at_cnn = timestamps[cnn_anchors]

    # Labels at CNN anchors (already in cnn_labels, but let's also get raw for MFE/MAE)
    # For MFE/MAE we need the label series around each entry
    # labels_Xs[anchor] gives the mid-price move in ticks over X seconds from that event

    return {
        "n_events": n_events,
        "n_preds": n_cnn,
        "cnn_preds": cnn_preds,      # (N, 3) - 1s, 5s, 10s
        "cnn_labels": cnn_labels,    # (N, 3) - ground truth moves in ticks
        "ptst_preds": ptst_at_cnn,   # (N, 3) - aligned PatchTST preds (NaN if missing)
        "vol_preds": vol_at_cnn,     # (N,) - aligned vol predictions (NaN if missing)
        "timestamps": ts_at_cnn,     # (N,) - nanosecond timestamps
        "cnn_anchors": cnn_anchors,  # (N,) - source event indices
        # Keep raw source labels for MFE/MAE lookups
        "src_labels_1s": labels_1s,
        "src_labels_5s": labels_5s,
        "src_labels_10s": labels_10s,
        "src_timestamps": timestamps,
    }


# ── MFE/MAE Computation ─────────────────────────────────────────────

def compute_mfe_mae(
    direction: int,
    entry_anchor: int,
    src_labels_1s: np.ndarray,
    src_labels_5s: np.ndarray,
    src_labels_10s: np.ndarray,
    timeout_s: float,
    src_timestamps: np.ndarray,
    entry_ts: int,
) -> Tuple[float, float]:
    """
    Estimate MFE and MAE in ticks for a trade.
    Uses 1s, 5s, 10s label snapshots as price path approximation.
    """
    # Get the mid-price moves at 1s, 5s, 10s from entry
    if entry_anchor >= len(src_labels_1s):
        return 0.0, 0.0

    moves = []
    move_1s = src_labels_1s[entry_anchor]
    move_5s = src_labels_5s[entry_anchor]
    move_10s = src_labels_10s[entry_anchor]

    if not np.isnan(move_1s):
        moves.append(direction * move_1s)
    if not np.isnan(move_5s):
        moves.append(direction * move_5s)
    if not np.isnan(move_10s):
        moves.append(direction * move_10s)

    if not moves:
        return 0.0, 0.0

    mfe = max(0.0, max(moves))
    mae = max(0.0, -min(moves))
    return mfe, mae


# ── Backtest Engine ──────────────────────────────────────────────────

def run_backtest_single_day(
    day_data: Dict,
    config_name: str,
    horizon: str,
    z_threshold: float,
    date_str: str,
    fold_idx: int,
    vol_p25: float,
    vol_p75: float,
) -> List[Trade]:
    """
    Run backtest for a single day with given config.

    For each prediction event:
    1. Compute z-score
    2. Check entry signal
    3. Apply adaptive TP/SL
    4. Determine exit via label-based PnL approximation
    """
    h_idx = HORIZON_IDX[horizon]
    cnn_preds_h = day_data["cnn_preds"][:, h_idx]
    cnn_labels_h = day_data["cnn_labels"][:, h_idx]
    ptst_preds_h = day_data["ptst_preds"][:, h_idx]
    vol_preds = day_data["vol_preds"]
    timestamps = day_data["timestamps"]
    cnn_anchors = day_data["cnn_anchors"]
    n = day_data["n_preds"]

    # Compute rolling z-scores for CNN predictions
    zscores = compute_zscore_rolling(cnn_preds_h)

    # Also compute PatchTST z-scores for confluence
    ptst_valid = ~np.isnan(ptst_preds_h)
    ptst_zscores = np.zeros(n)
    if np.any(ptst_valid):
        ptst_filled = np.where(ptst_valid, ptst_preds_h, 0.0)
        ptst_zscores = compute_zscore_rolling(ptst_filled)
        ptst_zscores[~ptst_valid] = 0.0

    trades = []
    i = 0

    while i < n:
        z = zscores[i]
        abs_z = abs(z)

        # Skip if below threshold
        if abs_z < z_threshold:
            i += 1
            continue

        direction = 1 if z > 0 else -1
        et_hour = ts_ns_to_et_hour(timestamps[i])
        session = get_session(et_hour)
        vol_class = "medium"
        if not np.isnan(vol_preds[i]):
            vol_class = classify_vol(vol_preds[i], vol_p25, vol_p75)

        # ── Entry Filters ────────────────────────────────────────
        if config_name == "D":
            # PatchTST confluence: both must agree on direction
            if not ptst_valid[i]:
                i += 1
                continue
            ptst_direction = 1 if ptst_preds_h[i] > 0 else -1
            if ptst_direction != direction:
                i += 1
                continue

        # ── Determine TP/SL/Timeout ──────────────────────────────
        if config_name == "A":
            params = TOD_TPSL[session]
            tp, sl, timeout_s = params["tp"], params["sl"], params["timeout_s"]
        elif config_name == "B":
            params = VOL_TPSL[vol_class]
            tp, sl, timeout_s = params["tp"], params["sl"], params["timeout_s"]
        elif config_name in ("C", "D"):
            # Combined: ToD base * vol multiplier
            base = TOD_TPSL[session]
            mult = VOL_MULTIPLIER[vol_class]
            tp = max(1, round(base["tp"] * mult))
            sl = max(1, round(base["sl"] * mult))
            timeout_s = max(5, base["timeout_s"] * mult)
        elif config_name == "E":
            tp, sl, timeout_s = STATIC_TP, STATIC_SL, STATIC_TIMEOUT_S
        else:
            tp, sl, timeout_s = STATIC_TP, STATIC_SL, STATIC_TIMEOUT_S

        # ── Simulate Trade Using Labels ──────────────────────────
        # We approximate by checking label moves at future prediction events
        # Each prediction event is CNN_STRIDE=500 source events apart
        # Approximate time between prediction events from timestamps
        entry_ts = timestamps[i]
        entry_anchor = cnn_anchors[i]

        exit_reason = "timeout"
        exit_idx = i
        pnl_gross = 0.0
        best_favorable = 0.0
        worst_adverse = 0.0

        # Look ahead through future prediction events
        for j in range(i + 1, min(i + 200, n)):  # cap lookahead
            elapsed_s = (timestamps[j] - entry_ts) / 1e9
            if elapsed_s < 0:
                continue

            # Approximate cumulative move from entry to event j
            # Use the label at entry for the closest horizon
            # For longer holds, we chain: move(entry->j) ~ sum of incremental label moves
            # But labels are absolute moves from each anchor, not incremental
            # Best approximation: use the label at entry point for the matching horizon
            # and the label at event j for a running estimate

            # Incremental move estimate: label_h at each intermediate step gives
            # the move over h seconds from that point. We want the cumulative move
            # from entry. Approximate: label at entry for elapsed time.
            if elapsed_s <= 1.0:
                move_ticks = day_data["src_labels_1s"][entry_anchor]
            elif elapsed_s <= 5.0:
                # Interpolate between 1s and 5s labels
                alpha = (elapsed_s - 1.0) / 4.0
                move_ticks = (
                    (1 - alpha) * day_data["src_labels_1s"][entry_anchor]
                    + alpha * day_data["src_labels_5s"][entry_anchor]
                )
            elif elapsed_s <= 10.0:
                alpha = (elapsed_s - 5.0) / 5.0
                move_ticks = (
                    (1 - alpha) * day_data["src_labels_5s"][entry_anchor]
                    + alpha * day_data["src_labels_10s"][entry_anchor]
                )
            else:
                # Beyond 10s: use 10s label as best estimate, then add incremental
                # from subsequent anchors
                move_ticks = day_data["src_labels_10s"][entry_anchor]
                # Add incremental moves from intermediate anchors
                for k in range(i + 1, j + 1):
                    if k >= n:
                        break
                    k_anchor = cnn_anchors[k]
                    k_elapsed = (timestamps[k] - timestamps[k-1]) / 1e9
                    if k_elapsed <= 1.0:
                        move_ticks += day_data["src_labels_1s"][cnn_anchors[k-1]]
                    elif k_elapsed <= 5.0:
                        move_ticks += day_data["src_labels_5s"][cnn_anchors[k-1]]
                    else:
                        move_ticks += day_data["src_labels_10s"][cnn_anchors[k-1]]
                    # Only accumulate once to avoid runaway
                    break

            if np.isnan(move_ticks):
                continue

            directional_move = direction * move_ticks
            best_favorable = max(best_favorable, directional_move)
            worst_adverse = min(worst_adverse, directional_move)

            # Check TP
            if directional_move >= tp:
                exit_reason = "tp"
                pnl_gross = tp  # capped at TP
                exit_idx = j
                break

            # Check SL
            if directional_move <= -sl:
                exit_reason = "sl"
                pnl_gross = -sl  # capped at SL
                exit_idx = j
                break

            # Config D: PatchTST reversal exit
            if config_name == "D" and ptst_valid[j]:
                ptst_dir_now = 1 if ptst_preds_h[j] > 0 else -1
                if ptst_dir_now != direction:
                    exit_reason = "ptst_reversal"
                    pnl_gross = directional_move
                    exit_idx = j
                    break

            # Timeout
            if elapsed_s >= timeout_s:
                exit_reason = "timeout"
                pnl_gross = directional_move
                exit_idx = j
                break
        else:
            # Exhausted lookahead without exit — use last known move
            if i + 1 < n:
                elapsed_s = (timestamps[min(i + 199, n - 1)] - entry_ts) / 1e9
                if elapsed_s <= 1.0:
                    move_ticks = day_data["src_labels_1s"][entry_anchor]
                elif elapsed_s <= 5.0:
                    move_ticks = day_data["src_labels_5s"][entry_anchor]
                else:
                    move_ticks = day_data["src_labels_10s"][entry_anchor]
                if not np.isnan(move_ticks):
                    pnl_gross = direction * move_ticks

        exit_ts = timestamps[exit_idx] if exit_idx < n else entry_ts
        hold_time = (exit_ts - entry_ts) / 1e9

        # MFE/MAE
        mfe = max(0.0, best_favorable)
        mae = max(0.0, -worst_adverse)

        pnl_net = pnl_gross - ROUND_TRIP_COST

        trades.append(Trade(
            entry_idx=i,
            exit_idx=exit_idx,
            direction=direction,
            entry_ts_ns=int(entry_ts),
            exit_ts_ns=int(exit_ts),
            hold_time_s=hold_time,
            pnl_ticks_gross=pnl_gross,
            pnl_ticks_net=pnl_net,
            exit_reason=exit_reason,
            session=session,
            mfe_ticks=mfe,
            mae_ticks=mae,
            confidence_z=abs_z,
            fold=fold_idx,
            date=date_str,
            horizon=horizon,
        ))

        # Skip ahead past exit to avoid overlapping trades
        i = exit_idx + 1
        continue

    return trades


def aggregate_results(
    trades: List[Trade],
    config_name: str,
    horizon: str,
    tier_name: str,
    tier_pct: float,
) -> ConfigResult:
    """Filter trades by confidence tier and compute aggregate stats."""

    if tier_pct > 0 and trades:
        z_vals = np.array([t.confidence_z for t in trades])
        threshold = np.percentile(z_vals, tier_pct * 100)
        filtered = [t for t in trades if t.confidence_z >= threshold]
    else:
        filtered = trades

    result = ConfigResult(
        config_name=config_name,
        horizon=horizon,
        tier=tier_name,
    )

    if not filtered:
        return result

    pnls = np.array([t.pnl_ticks_net for t in filtered])
    result.trade_count = len(filtered)
    result.total_pnl_ticks = float(np.sum(pnls))
    result.total_pnl_usd = result.total_pnl_ticks * TICK_VAL
    result.win_count = int(np.sum(pnls > 0))
    result.win_rate = result.win_count / result.trade_count if result.trade_count > 0 else 0
    result.sortino = compute_sortino(pnls)
    result.avg_hold_s = float(np.mean([t.hold_time_s for t in filtered]))
    result.avg_mfe = float(np.mean([t.mfe_ticks for t in filtered]))
    result.avg_mae = float(np.mean([t.mae_ticks for t in filtered]))

    exit_reasons = [t.exit_reason for t in filtered]
    n = len(filtered)
    result.tp_rate = exit_reasons.count("tp") / n
    result.sl_rate = exit_reasons.count("sl") / n
    result.timeout_rate = exit_reasons.count("timeout") / n
    result.ptst_reversal_rate = exit_reasons.count("ptst_reversal") / n

    # Per-session breakdown
    for sess_name, _, _ in SESSION_BOUNDS_ET:
        sess_trades = [t for t in filtered if t.session == sess_name]
        if not sess_trades:
            result.per_session[sess_name] = {
                "trade_count": 0, "total_pnl_ticks": 0, "win_rate": 0,
                "avg_hold_s": 0, "tp_rate": 0, "sl_rate": 0, "timeout_rate": 0,
            }
            continue
        s_pnls = np.array([t.pnl_ticks_net for t in sess_trades])
        s_exits = [t.exit_reason for t in sess_trades]
        sn = len(sess_trades)
        result.per_session[sess_name] = {
            "trade_count": sn,
            "total_pnl_ticks": float(np.sum(s_pnls)),
            "win_rate": float(np.sum(s_pnls > 0)) / sn,
            "avg_hold_s": float(np.mean([t.hold_time_s for t in sess_trades])),
            "tp_rate": s_exits.count("tp") / sn,
            "sl_rate": s_exits.count("sl") / sn,
            "timeout_rate": s_exits.count("timeout") / sn,
        }

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    print("=" * 80)
    print("ADAPTIVE TP/SL BACKTEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Discover dates with both CNN and PatchTST predictions
    date_map = discover_fold_date_mapping()
    print(f"\nFound {len(date_map)} common dates: {sorted(date_map.keys())}")

    if not date_map:
        print("ERROR: No common dates found between CNN Mamba v2 and PatchTST.")
        sys.exit(1)

    # ── Phase 1: Load all day data ───────────────────────────────
    print("\n── Loading predictions and source data ──")
    all_day_data = {}
    all_vol_preds = []

    for date_str, info in sorted(date_map.items()):
        print(f"  Loading {date_str}...", end=" ", flush=True)
        day_data = load_day_data(info)
        if day_data is None:
            print("SKIP")
            continue
        all_day_data[date_str] = day_data
        # Collect vol predictions for percentile computation
        valid_vol = day_data["vol_preds"][~np.isnan(day_data["vol_preds"])]
        all_vol_preds.extend(valid_vol.tolist())
        print(f"OK ({day_data['n_preds']} preds, "
              f"PatchTST coverage: {np.mean(~np.isnan(day_data['ptst_preds'][:, 0])):.1%})")

    if not all_day_data:
        print("ERROR: No valid day data loaded.")
        sys.exit(1)

    # Compute global vol percentiles for Config B/C/D
    if all_vol_preds:
        vol_arr = np.array(all_vol_preds)
        vol_p25 = float(np.percentile(vol_arr, 25))
        vol_p75 = float(np.percentile(vol_arr, 75))
        print(f"\nVol percentiles: P25={vol_p25:.4f}, P75={vol_p75:.4f}")
    else:
        vol_p25, vol_p75 = 0.0, 1.0
        print("\nWARN: No vol predictions available, using dummy percentiles")

    # ── Phase 2: Run backtests ───────────────────────────────────
    configs = ["A", "B", "C", "D", "E"]
    config_descriptions = {
        "A": "Time-of-Day Adaptive",
        "B": "Volatility-Adaptive",
        "C": "Combined ToD+Vol",
        "D": "PatchTST Confluence + Combined",
        "E": "Static Baseline",
    }

    # Z-score thresholds: we use a moderate threshold of 1.5 for entry
    Z_THRESHOLD = 1.5

    all_results = {}  # (config, horizon, tier) -> ConfigResult
    all_trades = {}   # (config, horizon) -> List[Trade]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config {config}: {config_descriptions[config]}")
        print(f"{'='*60}")

        for horizon in HORIZON_NAMES:
            print(f"\n  Horizon: {horizon}")
            trades_all = []

            for date_str, day_data in sorted(all_day_data.items()):
                cnn_fold = date_map[date_str]["cnn_fold"]
                day_trades = run_backtest_single_day(
                    day_data=day_data,
                    config_name=config,
                    horizon=horizon,
                    z_threshold=Z_THRESHOLD,
                    date_str=date_str,
                    fold_idx=cnn_fold,
                    vol_p25=vol_p25,
                    vol_p75=vol_p75,
                )
                trades_all.extend(day_trades)
                if day_trades:
                    day_pnl = sum(t.pnl_ticks_net for t in day_trades)
                    print(f"    {date_str}: {len(day_trades)} trades, "
                          f"PnL={day_pnl:+.1f}t")

            all_trades[(config, horizon)] = trades_all
            print(f"  Total: {len(trades_all)} trades")

            # Compute results per confidence tier
            for tier_name, tier_pct in CONFIDENCE_TIERS.items():
                result = aggregate_results(
                    trades_all, config, horizon, tier_name, tier_pct
                )
                all_results[(config, horizon, tier_name)] = result

    # ── Phase 3: Output Results ──────────────────────────────────
    elapsed = time.time() - start_time

    # Create output directory
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE / "output" / f"adaptive_tpsl_backtest_{ts_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Print Comparison Table ───────────────────────────────────
    print("\n" + "=" * 120)
    print("ADAPTIVE TP/SL BACKTEST RESULTS — CONFIG COMPARISON")
    print("=" * 120)
    print(f"Z-score entry threshold: {Z_THRESHOLD}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.3f} ticks "
          f"(commission={COMMISSION}t x2 + spread={SPREAD}t x2)")
    print(f"Dates: {sorted(all_day_data.keys())}")
    print(f"Elapsed: {elapsed:.1f}s")

    for horizon in HORIZON_NAMES:
        print(f"\n{'─'*120}")
        print(f"HORIZON: {horizon}")
        print(f"{'─'*120}")

        # Header
        header = f"{'Config':<35} {'Trades':>7} {'PnL(t)':>9} {'PnL($)':>10} " \
                 f"{'WinR':>6} {'Sortino':>8} {'AvgHold':>8} " \
                 f"{'TP%':>6} {'SL%':>6} {'TO%':>6} {'PtR%':>6} " \
                 f"{'MFE':>6} {'MAE':>6}"
        print(header)
        print("─" * 120)

        for tier_name in CONFIDENCE_TIERS:
            print(f"\n  ── {tier_name} ──")
            for config in configs:
                r = all_results.get((config, horizon, tier_name))
                if r is None or r.trade_count == 0:
                    print(f"  {config}: {config_descriptions[config]:<28} {'no trades':>7}")
                    continue
                label = f"{config}: {config_descriptions[config]}"
                print(f"  {label:<33} {r.trade_count:>7} {r.total_pnl_ticks:>+9.1f} "
                      f"{r.total_pnl_usd:>+10.0f} "
                      f"{r.win_rate:>6.1%} {r.sortino:>8.3f} {r.avg_hold_s:>7.1f}s "
                      f"{r.tp_rate:>6.1%} {r.sl_rate:>6.1%} {r.timeout_rate:>6.1%} "
                      f"{r.ptst_reversal_rate:>6.1%} "
                      f"{r.avg_mfe:>6.2f} {r.avg_mae:>6.2f}")

    # ── Per-Session Breakdown ────────────────────────────────────
    print("\n" + "=" * 120)
    print("PER-SESSION BREAKDOWN (All predictions tier, 10s horizon)")
    print("=" * 120)

    for config in configs:
        r = all_results.get((config, "10s", "All"))
        if r is None or r.trade_count == 0:
            continue
        print(f"\nConfig {config}: {config_descriptions[config]}")
        print(f"  {'Session':<15} {'Trades':>7} {'PnL(t)':>9} {'WinR':>6} "
              f"{'AvgHold':>8} {'TP%':>6} {'SL%':>6} {'TO%':>6}")
        print("  " + "─" * 80)
        for sess_name, _, _ in SESSION_BOUNDS_ET:
            s = r.per_session.get(sess_name, {})
            tc = s.get("trade_count", 0)
            if tc == 0:
                print(f"  {sess_name:<15} {0:>7}")
                continue
            print(f"  {sess_name:<15} {tc:>7} {s['total_pnl_ticks']:>+9.1f} "
                  f"{s['win_rate']:>6.1%} {s['avg_hold_s']:>7.1f}s "
                  f"{s['tp_rate']:>6.1%} {s['sl_rate']:>6.1%} {s['timeout_rate']:>6.1%}")

    # ── MFE/MAE Summary ─────────────────────────────────────────
    print("\n" + "=" * 120)
    print("MFE/MAE DISTRIBUTIONS (10s horizon, All tier)")
    print("=" * 120)

    for config in configs:
        trades = all_trades.get((config, "10s"), [])
        if not trades:
            continue
        mfes = np.array([t.mfe_ticks for t in trades])
        maes = np.array([t.mae_ticks for t in trades])
        print(f"\nConfig {config}: {config_descriptions[config]}")
        print(f"  MFE — mean={np.mean(mfes):.2f}, median={np.median(mfes):.2f}, "
              f"p75={np.percentile(mfes, 75):.2f}, p95={np.percentile(mfes, 95):.2f}")
        print(f"  MAE — mean={np.mean(maes):.2f}, median={np.median(maes):.2f}, "
              f"p75={np.percentile(maes, 75):.2f}, p95={np.percentile(maes, 95):.2f}")

    # ── Winner Summary ───────────────────────────────────────────
    print("\n" + "=" * 120)
    print("WINNER SUMMARY")
    print("=" * 120)

    for horizon in HORIZON_NAMES:
        print(f"\n  {horizon}:")
        for tier_name in ["All", "Top10%", "Top5%", "Top1%"]:
            best_config = None
            best_sortino = -999
            for config in configs:
                r = all_results.get((config, horizon, tier_name))
                if r and r.trade_count >= 10 and r.sortino > best_sortino:
                    best_sortino = r.sortino
                    best_config = config
            if best_config:
                r = all_results[(best_config, horizon, tier_name)]
                print(f"    {tier_name:<10} -> Config {best_config} "
                      f"({config_descriptions[best_config]}) "
                      f"Sortino={r.sortino:.3f}, PnL={r.total_pnl_ticks:+.1f}t, "
                      f"WinR={r.win_rate:.1%}, N={r.trade_count}")

    # ── Save JSON Results ────────────────────────────────────────
    results_json = {
        "metadata": {
            "timestamp": ts_str,
            "z_threshold": Z_THRESHOLD,
            "round_trip_cost_ticks": ROUND_TRIP_COST,
            "commission_ticks": COMMISSION,
            "spread_ticks": SPREAD,
            "tick_value_usd": TICK_VAL,
            "dates": sorted(all_day_data.keys()),
            "n_dates": len(all_day_data),
            "elapsed_s": elapsed,
            "vol_p25": vol_p25,
            "vol_p75": vol_p75,
            "configs": {
                "A": "Time-of-Day Adaptive TP/SL",
                "B": "Volatility-Adaptive TP/SL",
                "C": "Combined ToD+Vol Adaptive",
                "D": "PatchTST Confluence + Combined TP/SL",
                "E": "Static Baseline (TP=8, SL=5, TO=30s)",
            },
            "tod_tpsl": TOD_TPSL,
            "vol_tpsl": VOL_TPSL,
            "vol_multiplier": VOL_MULTIPLIER,
        },
        "results": {},
    }

    for (config, horizon, tier_name), r in all_results.items():
        key = f"{config}_{horizon}_{tier_name}"
        results_json["results"][key] = {
            "config": r.config_name,
            "horizon": r.horizon,
            "tier": r.tier,
            "trade_count": r.trade_count,
            "total_pnl_ticks": round(r.total_pnl_ticks, 3),
            "total_pnl_usd": round(r.total_pnl_usd, 2),
            "win_rate": round(r.win_rate, 4),
            "sortino": round(r.sortino, 4),
            "avg_hold_s": round(r.avg_hold_s, 2),
            "tp_rate": round(r.tp_rate, 4),
            "sl_rate": round(r.sl_rate, 4),
            "timeout_rate": round(r.timeout_rate, 4),
            "ptst_reversal_rate": round(r.ptst_reversal_rate, 4),
            "avg_mfe": round(r.avg_mfe, 3),
            "avg_mae": round(r.avg_mae, 3),
            "per_session": r.per_session,
        }

    # Save trade-level data for further analysis
    trades_json = {}
    for (config, horizon), trades in all_trades.items():
        key = f"{config}_{horizon}"
        trades_json[key] = [
            {
                "entry_idx": t.entry_idx,
                "exit_idx": t.exit_idx,
                "direction": t.direction,
                "hold_time_s": round(t.hold_time_s, 3),
                "pnl_gross": round(t.pnl_ticks_gross, 3),
                "pnl_net": round(t.pnl_ticks_net, 3),
                "exit_reason": t.exit_reason,
                "session": t.session,
                "mfe": round(t.mfe_ticks, 3),
                "mae": round(t.mae_ticks, 3),
                "confidence_z": round(t.confidence_z, 4),
                "fold": t.fold,
                "date": t.date,
            }
            for t in trades
        ]

    results_path = out_dir / "results.json"
    trades_path = out_dir / "trades.json"

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    with open(trades_path, "w") as f:
        json.dump(trades_json, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"  results.json: aggregate metrics per config/horizon/tier")
    print(f"  trades.json:  trade-level detail for all configs")
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
