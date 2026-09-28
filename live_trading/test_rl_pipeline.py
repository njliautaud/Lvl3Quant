#!/usr/bin/env python3
"""
test_rl_pipeline.py — End-to-End Validation of the Live RL Trading Pipeline
=============================================================================

Replays a recorded MBO day through the full stack:
  MBO events → CNN-Mamba v2 predictions → PatchTST predictions (optional)
            → RLExecutionAgent + RLPaperTradingWrapper → paper P&L

Run as:
    python3 test_rl_pipeline.py
    python3 test_rl_pipeline.py --weights /path/to/checkpoint.pt
    python3 test_rl_pipeline.py --data-dir /path/to/mbo_events --pred-dir /path/to/cm_preds

Exit codes:
    0  — all assertions passed
    1  — fatal error (missing data, NaN obs, etc.)
    2  — non-fatal warning (pipeline ran but had errors logged)

Author: Claude (Infrastructure Builder)
Date:   2026-05-03
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("test_rl_pipeline")

# ── Default paths (canonical layout) ─────────────────────────────────────────
LVL3 = Path("/home/jupiter/Lvl3Quant")
DEFAULT_DATA_DIR   = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
DEFAULT_PRED_DIR   = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar"
DEFAULT_PST_DIR    = LVL3 / "output" / "patchtst_smart_v3_mar"
DEFAULT_WEIGHTS    = LVL3 / "output" / "fifo_rl_test" / "best_agent_ppo_20260502_183727.pt"

# ── Constants (must match fifo_rl_env.py exactly) ────────────────────────────
PRED_STRIDE = 50
PRED_WINDOW = 1000

# Event type codes
EVENT_TRADE  = 0
EVENT_CANCEL = 3

# Feature column indices
COL_SIDE      = 2
COL_PRICE_REL = 3   # price_rel_ticks
COL_QTY_LOG   = 4
COL_SPREAD    = 5

# ES constants
TICK_VALUE = 12.50
TICK_SIZE  = 0.25

# US DST 2026 transitions (hard-coded for speed)
_DST_START_2026 = datetime.date(2026, 3, 8)   # second Sunday March
_DST_END_2026   = datetime.date(2026, 11, 1)  # first Sunday November


# ══════════════════════════════════════════════════════════════════════════════
# Timezone / RTH helpers (correct for multi-day MBO files)
# ══════════════════════════════════════════════════════════════════════════════

def _et_utc_offset_s(year: int, month: int, day: int) -> int:
    """Return ET→UTC offset in seconds for a given date (4h EDT or 5h EST)."""
    d = datetime.date(year, month, day)
    if _DST_START_2026 <= d < _DST_END_2026:
        return 4 * 3600   # EDT = UTC-4
    return 5 * 3600       # EST = UTC-5


def _rth_ns_bounds(date_str: str) -> Tuple[int, int]:
    """
    Return (rth_start_ns, rth_end_ns) for RTH on the given YYYYMMDD date.

    RTH = 09:30–16:00 ET.  Timestamps are nanoseconds since Unix epoch (UTC).

    Uses calendar.timegm (UTC-based, no local-timezone pollution) plus the
    correct EST/EDT offset for the specific date.
    """
    yr, mo, dy = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    et_off = _et_utc_offset_s(yr, mo, dy)   # seconds: positive = hours behind UTC

    # Build naive local ET datetime, then convert to UTC epoch seconds
    rth_start_local = datetime.datetime(yr, mo, dy, 9, 30, 0)
    rth_end_local   = datetime.datetime(yr, mo, dy, 16, 0, 0)

    # calendar.timegm interprets the tuple as UTC
    rth_start_utc_s = calendar.timegm(rth_start_local.timetuple()) + et_off
    rth_end_utc_s   = calendar.timegm(rth_end_local.timetuple())   + et_off

    return rth_start_utc_s * 1_000_000_000, rth_end_utc_s * 1_000_000_000


def _binary_search_ts(timestamps: np.ndarray, target_ns: int) -> int:
    """Return first index where timestamps[i] >= target_ns (or len if none)."""
    lo, hi = 0, len(timestamps) - 1
    result = len(timestamps)
    while lo <= hi:
        mid = (lo + hi) // 2
        if timestamps[mid] >= target_ns:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return result


def _find_rth_bounds(timestamps: np.ndarray, date_str: str) -> Tuple[int, int]:
    """
    Return (rth_start_idx, rth_end_idx) within the timestamps array for the
    named trading date's RTH window (09:30–16:00 ET).

    The MBO files can contain events from multiple calendar days (sliding
    window data), so we match against absolute UTC timestamps, not TOD.
    """
    rth_start_ns, rth_end_ns = _rth_ns_bounds(date_str)
    start_idx = _binary_search_ts(timestamps, rth_start_ns)
    end_idx   = _binary_search_ts(timestamps, rth_end_ns)
    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════════════
# Prediction index helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_pred_index(pred_dir: Path) -> Dict[str, Path]:
    """Build date→fold_file index by scanning oot_files in each fold."""
    index: Dict[str, Path] = {}
    for fold_f in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        try:
            d = np.load(fold_f, allow_pickle=True)
            for of in (d.get("oot_files") or []):
                m = re.search(r"(\d{8})", str(of))
                if m:
                    dt = m.group(1)
                    if dt not in index:
                        index[dt] = fold_f
        except Exception:
            continue
    return index


def _pick_test_date(
    data_dir: Path,
    cm_index: Dict[str, Path],
    pst_index: Optional[Dict[str, Path]],
) -> Tuple[str, Path, Path, Optional[Path]]:
    """
    Pick the most recent date that has:
      - An MBO events file
      - CNN-Mamba predictions
    Prefer dates that also have PatchTST predictions.
    """
    mbo_files = sorted(data_dir.glob("2026????_mbo_events.npz"))
    if not mbo_files:
        raise FileNotFoundError(f"No MBO event files found in {data_dir}")

    candidate_date = None
    candidate_mbo  = None
    candidate_cm   = None

    for mbo_f in reversed(mbo_files):
        dt = mbo_f.stem[:8]
        if dt not in cm_index:
            continue
        pst_f = pst_index.get(dt) if pst_index else None
        if pst_f is not None:
            return dt, mbo_f, cm_index[dt], pst_f
        if candidate_date is None:
            candidate_date = dt
            candidate_mbo  = mbo_f
            candidate_cm   = cm_index[dt]

    if candidate_date is None:
        raise RuntimeError("No MBO date with CNN-Mamba predictions found.")
    return candidate_date, candidate_mbo, candidate_cm, None


# ══════════════════════════════════════════════════════════════════════════════
# Market state extractor
# ══════════════════════════════════════════════════════════════════════════════

def _build_market_state(
    evt: np.ndarray,
    et: int,
    ts_ns: int,
    mid_price_usd: float,
) -> Dict[str, Any]:
    """
    Build market_state_dict for RLExecutionAgent.update_state() from one MBO row.

    The MBO feature col[3] is price_rel_ticks (relative to a running mid).
    We maintain a running absolute mid externally and use that to derive
    bid/ask prices for the wrapper's P&L accounting.
    """
    spread_ticks = float(evt[COL_SPREAD])
    half_spread  = max(spread_ticks / 2.0, 0.5)
    price_rel    = float(evt[COL_PRICE_REL])
    qty          = math.exp(float(evt[COL_QTY_LOG]))
    side         = float(evt[COL_SIDE])

    bid_pts = mid_price_usd - half_spread * TICK_SIZE
    ask_pts = mid_price_usd + half_spread * TICK_SIZE

    is_trade  = (et == EVENT_TRADE)
    is_cancel = (et == EVENT_CANCEL)

    return {
        "timestamp_s":    ts_ns / 1e9,
        "mid_price":      mid_price_usd,
        "best_bid":       bid_pts,
        "best_ask":       ask_pts,
        "spread_ticks":   spread_ticks,
        "price_rel_ticks":price_rel,
        "is_trade":       is_trade,
        "trade_side":     int(side) if is_trade else 0,
        "trade_qty":      qty       if is_trade else 0.0,
        "is_cancel":      is_cancel,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Observation validator
# ══════════════════════════════════════════════════════════════════════════════

def _check_obs_valid(obs: np.ndarray, step_i: int, errors: List[str]) -> bool:
    if obs.shape != (48,):
        errors.append(f"step {step_i}: obs shape {obs.shape} != (48,)")
        return False
    if np.any(np.isnan(obs)):
        nan_dims = np.where(np.isnan(obs))[0].tolist()
        errors.append(f"step {step_i}: NaN in obs dims {nan_dims}")
        return False
    if np.any(np.isinf(obs)):
        inf_dims = np.where(np.isinf(obs))[0].tolist()
        errors.append(f"step {step_i}: Inf in obs dims {inf_dims}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Smart checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint_safe(
    policy,
    weights_path: str,
    device,
    hidden_dim: int,
    obs_dim: int,
) -> Tuple[bool, str]:
    """
    Attempt to load checkpoint into policy.  If there's an OBS_DIM mismatch
    (e.g. checkpoint was trained with OBS_DIM=45, current model=48), log a
    clear warning and return (False, reason) so the caller can fall back to
    random weights rather than crashing.

    Returns (loaded: bool, message: str).
    """
    import torch

    path = Path(weights_path)
    if not path.exists():
        return False, f"checkpoint not found: {path}"

    ckpt = torch.load(str(path), map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        return False, f"unrecognised checkpoint format in {path.name}"

    # Check for size mismatch before attempting load
    model_state = policy.state_dict()
    mismatches = []
    for key, val in state_dict.items():
        if key in model_state and model_state[key].shape != val.shape:
            mismatches.append(
                f"  {key}: checkpoint {tuple(val.shape)} vs model {tuple(model_state[key].shape)}"
            )

    if mismatches:
        ckpt_obs_dim = None
        # Try to infer the checkpoint OBS_DIM from input_norm.weight
        if "input_norm.weight" in state_dict:
            ckpt_obs_dim = state_dict["input_norm.weight"].shape[0]
        reason = (
            f"OBS_DIM mismatch: checkpoint has obs_dim={ckpt_obs_dim}, "
            f"current model has obs_dim={obs_dim}.  "
            f"This checkpoint was likely trained before the alpha-awareness "
            f"features (obs[45:48]) were added.  "
            f"Falling back to RANDOM WEIGHTS — pipeline plumbing test only.\n"
            + "\n".join(mismatches)
        )
        return False, reason

    try:
        policy.load_state_dict(state_dict)
        meta_parts = []
        for k in ("fold", "epoch", "best_sortino"):
            if isinstance(ckpt, dict) and k in ckpt:
                meta_parts.append(f"{k}={ckpt[k]}")
        return True, f"loaded {path.name}" + (f"  [{', '.join(meta_parts)}]" if meta_parts else "")
    except Exception as exc:
        return False, f"load_state_dict failed: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# Main test runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_test(
    data_dir:        Path,
    pred_dir:        Path,
    patchtst_dir:    Optional[Path],
    weights_path:    Optional[Path],
    max_events:      int   = 200_000,
    action_stride:   int   = PRED_STRIDE,
    hidden_dim:      int   = 256,
    device:          str   = "cpu",
    verbose:         bool  = False,
) -> int:
    """
    Run the full pipeline test.
    Returns exit code: 0=pass, 1=fatal error, 2=warnings only.
    """
    print("=" * 72)
    print("  Live RL Trading Pipeline — End-to-End Validation Test")
    print("=" * 72)

    # ── 1. Check imports ──────────────────────────────────────────────────────
    print("\n[1/7] Checking imports...")
    try:
        import torch
        print(f"      PyTorch {torch.__version__} available")
    except ImportError:
        print("      WARNING: PyTorch not installed — RL agent will use random init.")

    module_path = Path(__file__).parent / "rl_execution_agent.py"
    if not module_path.exists():
        print(f"      FATAL: rl_execution_agent.py not found at {module_path}")
        return 1

    sys.path.insert(0, str(module_path.parent))
    try:
        from rl_execution_agent import (
            RLAgentConfig,
            RLExecutionAgent,
            RLPaperTradingWrapper,
            FIFOActorCritic,
            ACTION_NAMES,
            OBS_DIM,
            N_ACTIONS,
            TICK_VALUE as RL_TICK_VALUE,
        )
        print(f"      rl_execution_agent loaded — OBS_DIM={OBS_DIM}, N_ACTIONS={N_ACTIONS}")
    except ImportError as e:
        print(f"      FATAL: Could not import rl_execution_agent: {e}")
        return 1

    if OBS_DIM != 48:
        print(f"      FATAL: OBS_DIM={OBS_DIM} != 48 in rl_execution_agent.py")
        return 1
    if abs(RL_TICK_VALUE - TICK_VALUE) > 0.001:
        print(f"      FATAL: TICK_VALUE mismatch: module={RL_TICK_VALUE} vs test={TICK_VALUE}")
        return 1
    print("      OBS_DIM=48 confirmed. Cost constants match.")

    # ── 2. Discover data files ─────────────────────────────────────────────────
    print("\n[2/7] Discovering data files...")
    if not data_dir.exists():
        print(f"      FATAL: MBO data directory not found: {data_dir}")
        return 1
    if not pred_dir.exists():
        print(f"      FATAL: CNN-Mamba prediction directory not found: {pred_dir}")
        return 1

    cm_index  = _build_pred_index(pred_dir)
    pst_index = _build_pred_index(patchtst_dir) if (patchtst_dir and patchtst_dir.exists()) else None

    print(f"      CNN-Mamba index: {len(cm_index)} dates from {pred_dir.name}")
    if pst_index is not None:
        print(f"      PatchTST  index: {len(pst_index)} dates from {patchtst_dir.name}")
    else:
        print("      PatchTST: directory not found — skipping")

    try:
        test_date, mbo_file, cm_file, pst_file = _pick_test_date(data_dir, cm_index, pst_index)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"      FATAL: {e}")
        return 1

    print(f"      Selected test date: {test_date}")
    print(f"      MBO file:           {mbo_file.name}")
    print(f"      CNN-Mamba file:     {cm_file.name}")
    if pst_file:
        print(f"      PatchTST file:      {pst_file.name}")
    else:
        print("      PatchTST file:      NOT AVAILABLE for this date")

    # ── 3. Load data ──────────────────────────────────────────────────────────
    print("\n[3/7] Loading data...")
    t0 = time.time()

    try:
        mbo_data  = np.load(mbo_file)
        events    = mbo_data["events"]           # (N, 25) float32
        et_raw    = mbo_data["event_type_raw"]   # (N,) int8
        timestamps = mbo_data["timestamps"]      # (N,) int64 nanoseconds
    except Exception as e:
        print(f"      FATAL: Could not load MBO file: {e}")
        return 1

    n_events_total = len(events)
    print(f"      MBO events (file total): {n_events_total:,}")

    # Locate RTH window using absolute UTC timestamps (handles multi-day files)
    rth_start, rth_end = _find_rth_bounds(timestamps, test_date)
    rth_count = rth_end - rth_start

    if rth_count < 1000:
        print(f"      WARNING: Only {rth_count} RTH events found for {test_date}. "
              f"File may not cover this date's RTH. Attempting full-file replay.")
        rth_start = 0
        rth_end   = min(n_events_total, max_events * 2)
        rth_count = rth_end

    # Describe RTH window in human-readable UTC
    ts_start_utc = datetime.datetime.utcfromtimestamp(timestamps[rth_start] / 1e9)
    ts_end_idx   = min(rth_end, n_events_total) - 1
    ts_end_utc   = datetime.datetime.utcfromtimestamp(timestamps[ts_end_idx] / 1e9)
    print(f"      RTH window [{rth_start:,} → {rth_end:,}]: {rth_count:,} events")
    print(f"      UTC range: {ts_start_utc.strftime('%H:%M:%S')} → {ts_end_utc.strftime('%H:%M:%S')} "
          f"on {ts_start_utc.date()}")

    # Load CNN-Mamba predictions
    try:
        cm_data      = np.load(cm_file, allow_pickle=True)
        cm_preds     = cm_data["predictions"].astype(np.float32)    # (M, 3)
        cm_labels    = cm_data["labels"].astype(np.float32)         # (M, 3)
        cm_embeddings = (
            cm_data["embeddings"].astype(np.float32)
            if "embeddings" in cm_data else None
        )
        n_cm = len(cm_preds)
    except Exception as e:
        print(f"      FATAL: Could not load CNN-Mamba predictions: {e}")
        return 1

    print(f"      CNN-Mamba predictions: {n_cm:,}  shape={cm_preds.shape}  "
          f"range=[{cm_preds.min():.3f}, {cm_preds.max():.3f}]")
    if "ic_1s" in cm_data:
        print(f"      CNN-Mamba IC_1s (fold): {float(cm_data['ic_1s']):.4f}")
    if cm_embeddings is not None:
        print(f"      CNN-Mamba embeddings:   {cm_embeddings.shape}")

    # PatchTST predictions (optional)
    pst_preds = None
    n_pst     = 0
    if pst_file is not None:
        try:
            pst_data  = np.load(pst_file, allow_pickle=True)
            pst_preds = pst_data["predictions"].astype(np.float32)
            n_pst     = len(pst_preds)
            print(f"      PatchTST predictions:   {n_pst:,}  shape={pst_preds.shape}  "
                  f"range=[{pst_preds.min():.3f}, {pst_preds.max():.3f}]")
        except Exception as e:
            print(f"      WARNING: Could not load PatchTST predictions: {e}")
            pst_preds = None

    load_ms = (time.time() - t0) * 1000
    print(f"      Data loaded in {load_ms:.0f}ms")

    # ── 4. Build PCA for embeddings ───────────────────────────────────────────
    pca_components = None
    if cm_embeddings is not None:
        print("\n[4/7] Fitting PCA on CNN-Mamba embeddings...")
        E     = cm_embeddings.shape[1]
        n_comp = min(5, E)
        n_samp = min(5000, len(cm_embeddings))
        idx   = np.random.default_rng(42).choice(len(cm_embeddings), n_samp, replace=False)
        X     = cm_embeddings[idx].astype(np.float64)
        X    -= X.mean(axis=0)
        Q     = np.random.default_rng(42).standard_normal((E, n_comp))
        for _ in range(5):
            Q, _ = np.linalg.qr(X.T @ (X @ Q))
        _, _, Vt    = np.linalg.svd(X @ Q, full_matrices=False)
        pca_components = (Q @ Vt).T   # (n_comp, E)
        print(f"      PCA fitted: {E}D → {n_comp}D (on {n_samp} samples)")
    else:
        print("\n[4/7] No embeddings in CNN-Mamba fold — skipping PCA.")

    # ── 5. Initialise RL agent ────────────────────────────────────────────────
    print("\n[5/7] Initialising RL agent...")
    warnings: List[str] = []
    weights_loaded = False
    checkpoint_note = ""

    # Resolve weights
    wp_str = str(weights_path) if (weights_path and weights_path.exists()) else ""
    if not wp_str and weights_path:
        msg = f"Checkpoint not found at {weights_path}"
        print(f"      NOTE: {msg}")
        warnings.append(msg)

    cfg = RLAgentConfig(
        weights_path           = "",          # we load manually below for better error handling
        hidden_dim             = hidden_dim,
        device                 = device,
        greedy                 = True,
        use_patchtst           = (pst_preds is not None),
        use_vol_model          = True,
        use_embeddings         = (pca_components is not None),
        max_daily_loss_usd     = 10_000.0,   # wide — don't circuit-break during test
        max_consecutive_losses = 999,        # don't halt in test
        log_every_action       = False,
    )

    try:
        agent   = RLExecutionAgent(cfg)
        wrapper = RLPaperTradingWrapper(
            agent           = agent,
            config          = cfg,
            symbol          = "ESM6",
            max_spread_ticks= 8.0,           # wide — don't filter actions during test
        )
    except Exception as e:
        print(f"      FATAL: Could not initialise RL agent/wrapper: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Now attempt checkpoint load (with graceful mismatch handling)
    if wp_str:
        try:
            import torch
            _dev = torch.device(device if torch.cuda.is_available() else "cpu")
            loaded, msg = _load_checkpoint_safe(
                agent._policy, wp_str, _dev, hidden_dim, OBS_DIM
            )
            if loaded:
                agent._policy.eval()
                weights_loaded = True
                checkpoint_note = msg
                print(f"      Checkpoint: {msg}")
            else:
                warnings.append(f"Checkpoint not loaded: {msg}")
                print(f"      WARNING: {msg}")
                print("      Continuing with RANDOM weights (tests pipeline plumbing only).")
        except Exception as e:
            msg = f"Checkpoint load failed unexpectedly: {e}"
            warnings.append(msg)
            print(f"      WARNING: {msg}")
    else:
        print("      Using randomly initialised model (pipeline plumbing test only).")

    print("      Agent and wrapper ready.")
    print(f"      use_patchtst={cfg.use_patchtst}  "
          f"use_embeddings={cfg.use_embeddings}  "
          f"use_vol_model={cfg.use_vol_model}")

    # ── 6. Replay loop ────────────────────────────────────────────────────────
    n_replay   = min(max_events, rth_count)
    replay_end = rth_start + n_replay

    print(f"\n[6/7] Replaying {n_replay:,} RTH events (stride={action_stride})...")
    print(f"      Expected action calls: ~{n_replay // action_stride:,}")

    # Tracking
    errors: List[str]  = []
    action_counts      = Counter()
    latencies_ms: List[float] = []
    step_actions = 0
    obs_checks   = 0
    last_cm_idx  = 0

    # Running absolute mid price (approximate ES level)
    running_mid = 5250.0

    t_loop_start = time.time()
    t_progress   = t_loop_start

    for i in range(rth_start, replay_end):
        raw_i = i - rth_start   # 0-based within RTH
        evt   = events[i]
        et    = int(et_raw[i])
        ts_ns = int(timestamps[i])

        # Gently track price from price_rel_ticks (col 3)
        price_rel    = float(evt[COL_PRICE_REL])
        running_mid += price_rel * TICK_SIZE * 0.005   # tiny nudge

        # Only call the agent at prediction stride boundaries
        if raw_i % action_stride != 0:
            continue

        # Compute CNN-Mamba prediction index (mirrors fifo_rl_env.py mapping)
        pred_idx_raw = (raw_i - PRED_WINDOW + 1) // PRED_STRIDE
        cm_idx       = min(max(0, pred_idx_raw), n_cm - 1)
        last_cm_idx  = cm_idx

        # Fractional PatchTST alignment (PST uses a different stride/window)
        pst_idx = int(cm_idx * n_pst / max(n_cm, 1)) if pst_preds is not None else 0
        pst_idx = min(pst_idx, n_pst - 1)

        # Build prediction dict
        cm_pred = cm_preds[cm_idx]
        pred_dict: Dict[str, Any] = {
            "pred_1s":  float(cm_pred[0]),
            "pred_5s":  float(cm_pred[1]),
            "pred_10s": float(cm_pred[2]),
        }
        if pst_preds is not None:
            pst = pst_preds[pst_idx]
            pred_dict["patchtst_pred_1s"]  = float(pst[0])
            pred_dict["patchtst_pred_5s"]  = float(pst[1])
            pred_dict["patchtst_pred_10s"] = float(pst[2])
        if pca_components is not None and cm_idx < len(cm_embeddings):
            emb = cm_embeddings[cm_idx].astype(np.float64)
            pred_dict["pca_embedding"] = (pca_components @ emb).astype(np.float32)

        # Build market state dict
        mkt_dict = _build_market_state(evt, et, ts_ns, running_mid)

        # ── Call the RL pipeline ────────────────────────────────────────────
        t_act = time.perf_counter()
        try:
            result = wrapper.on_event(pred_dict, mkt_dict)
        except Exception as exc:
            errors.append(
                f"step {raw_i}: on_event raised {type(exc).__name__}: {exc}"
            )
            continue
        latencies_ms.append((time.perf_counter() - t_act) * 1000.0)

        action_counts[result.get("action_name", "unknown")] += 1
        step_actions += 1

        # Spot-check obs every 1000 actions (or first 5)
        if step_actions <= 5 or step_actions % 1000 == 0:
            try:
                obs_now = agent.build_obs()
                _check_obs_valid(obs_now, raw_i, errors)
                obs_checks += 1
            except Exception as exc:
                errors.append(f"step {raw_i}: build_obs raised: {exc}")

        if verbose and result["executed"] and result["action_name"] != "hold":
            log.info(
                "step=%d  action=%-18s executed=%s  pos=%d  day_pnl=$%+.2f",
                raw_i, result["action_name"], result["executed"],
                result["position"], result["daily_pnl_usd"],
            )

        # Progress every 5 seconds
        now = time.time()
        if now - t_progress > 5.0:
            pct  = 100.0 * raw_i / n_replay
            ela  = now - t_loop_start
            rate = raw_i / ela if ela > 0 else 1
            eta  = (n_replay - raw_i) / rate if rate > 0 else 0
            print(
                f"      [{pct:5.1f}%] {raw_i:>9,}/{n_replay:,}  "
                f"actions={step_actions:,}  trades={result['n_trades']}  "
                f"day_pnl=${result['daily_pnl_usd']:+.2f}  "
                f"rate={rate/1000:.0f}k ev/s  eta={eta:.0f}s"
            )
            t_progress = now

    loop_elapsed = time.time() - t_loop_start
    print(f"      Replay complete in {loop_elapsed:.1f}s")

    # ── 7. Report ─────────────────────────────────────────────────────────────
    summary     = wrapper.session_summary()
    agent_stats = summary.get("agent_stats", {})
    trades      = wrapper._trades
    n_trades    = summary["n_trades"]

    nan_count = inf_count = 0
    obs_range = "n/a"
    try:
        final_obs = agent.build_obs()
        nan_count = int(np.isnan(final_obs).sum())
        inf_count = int(np.isinf(final_obs).sum())
        obs_range = f"[{final_obs.min():.3f}, {final_obs.max():.3f}]"
    except Exception as e:
        errors.append(f"Final build_obs failed: {e}")

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    # ── Event replay stats ─────────────────────────────────────────────────────
    print(f"\n  EVENT REPLAY")
    print(f"    Date replayed:          {test_date}")
    print(f"    RTH events available:   {rth_count:,}")
    print(f"    Events replayed:        {n_replay:,}")
    print(f"    Replay rate:            {n_replay / loop_elapsed / 1000:.0f}k events/s")
    print(f"    CNN-Mamba preds used:   up to index {last_cm_idx:,} of {n_cm-1:,}")
    if pst_preds is not None:
        print(f"    PatchTST preds used:    YES  ({n_pst:,} total, fractional index align)")
    else:
        print(f"    PatchTST preds used:    NO (not available)")
    if weights_loaded:
        print(f"    Checkpoint:             LOADED — {checkpoint_note}")
    else:
        print(f"    Checkpoint:             RANDOM WEIGHTS (pipeline test only)")

    # ── RL action distribution ─────────────────────────────────────────────────
    print(f"\n  RL AGENT ACTIONS")
    total_acts = sum(action_counts.values())
    print(f"    Total action calls:     {step_actions:,}  (1 per {action_stride} events)")
    print(f"    Agent n_acts:           {agent_stats.get('n_acts', '?')}")
    print(f"    Action breakdown:")
    for aname in ["hold", "limit_buy_bid", "limit_sell_ask",
                  "market_buy", "market_sell", "cancel_order", "close_position"]:
        n = action_counts.get(aname, 0)
        pct = 100.0 * n / max(total_acts, 1)
        print(f"      {aname:<20s}: {n:6,}  ({pct:5.1f}%)")

    # ── Paper trading results ──────────────────────────────────────────────────
    print(f"\n  PAPER TRADING RESULTS")
    print(f"    Trades completed:       {n_trades}")
    if n_trades > 0:
        pnls_t = [t.pnl_ticks for t in trades]
        pnls_u = [t.pnl_usd   for t in trades]
        holds  = [t.hold_secs for t in trades]
        dirs   = Counter("LONG" if t.direction > 0 else "SHORT" for t in trades)
        exits  = Counter(t.exit_reason for t in trades)

        print(f"    Win rate:               {summary['win_rate']:.1%}")
        print(f"    Sortino (rolling):      {summary['sortino']:.4f}")
        print(f"    Profit factor:          {summary['profit_factor']:.4f}")
        print(f"    Total P&L:              {sum(pnls_t):+.3f} ticks  (${sum(pnls_u):+.2f})")
        print(f"    Avg P&L / trade:        {np.mean(pnls_t):+.3f} ticks  (${np.mean(pnls_u):+.2f})")
        print(f"    Avg hold time:          {np.mean(holds):.2f}s  (max {max(holds):.1f}s)")
        print(f"    Direction breakdown:    {dict(dirs)}")
        print(f"    Exit reasons:           {dict(exits)}")
        if trades:
            print(f"\n    First {min(10, n_trades)} trades:")
            print(f"      {'#':>3}  {'Dir':<5}  {'Entry':>8}  {'Exit':>8}  "
                  f"{'PnL(t)':>8}  {'PnL($)':>8}  {'Hold':>6}  {'Reason'}")
            for j, t in enumerate(trades[:10]):
                print(f"      {j+1:>3}  {'LONG' if t.direction>0 else 'SHORT':<5}  "
                      f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
                      f"{t.pnl_ticks:>+8.3f}  {t.pnl_usd:>+8.2f}  "
                      f"{t.hold_secs:>5.1f}s  {t.exit_reason}")
    else:
        print("    (No trades completed — expected with random/untrained weights)")
        print("    The pipeline processed events and generated actions; ")
        print("    P&L will be meaningful once real checkpoint is loaded.")

    # ── Inference latency ──────────────────────────────────────────────────────
    print(f"\n  INFERENCE LATENCY")
    if latencies_ms:
        lat = np.array(latencies_ms)
        print(f"    Samples:                {len(lat):,}")
        print(f"    Mean:                   {lat.mean():.3f}ms")
        print(f"    Median:                 {np.median(lat):.3f}ms")
        print(f"    p95:                    {np.percentile(lat, 95):.3f}ms")
        print(f"    p99:                    {np.percentile(lat, 99):.3f}ms")
        print(f"    Max:                    {lat.max():.3f}ms")
        budget_ms   = 250.0   # signal half-life
        over_budget = int((lat > budget_ms).sum())
        if over_budget:
            print(f"    WARNING: {over_budget} calls exceeded {budget_ms}ms budget")
        else:
            print(f"    All calls within {budget_ms}ms signal half-life budget.")

    # ── Observation health ─────────────────────────────────────────────────────
    print(f"\n  OBSERVATION HEALTH")
    print(f"    Spot-checks performed:  {obs_checks}")
    print(f"    NaN obs errors:         {len([e for e in errors if 'NaN' in e or 'nan' in e])}")
    print(f"    Final obs shape:        (48,)  NaN={nan_count}  Inf={inf_count}")
    print(f"    Final obs range:        {obs_range}")

    # ── Errors / warnings ─────────────────────────────────────────────────────
    all_issues = errors + warnings
    print(f"\n  ERRORS / WARNINGS ({len(all_issues)} total)")
    if all_issues:
        for msg in all_issues[:30]:
            tag = "WARN" if msg in warnings else "ERR "
            print(f"    [{tag}] {msg}")
        if len(all_issues) > 30:
            print(f"    ... and {len(all_issues) - 30} more")
    else:
        print("    None.")

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n  ASSERTIONS")
    n_pass = n_fail = 0

    def chk(label: str, ok: bool, detail: str = "") -> None:
        nonlocal n_pass, n_fail
        suffix = f"  [{detail}]" if detail else ""
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}{suffix}")
        if ok: n_pass += 1
        else:  n_fail += 1

    chk("OBS_DIM == 48",                    OBS_DIM == 48)
    chk("TICK_VALUE == 12.50",              abs(RL_TICK_VALUE - 12.50) < 0.001)
    chk("All action names are valid",
        all(k in ACTION_NAMES.values() for k in action_counts),
        str(set(action_counts) - set(ACTION_NAMES.values()) or "OK"))
    expected_min_acts = max(1, n_replay // action_stride - 5)
    chk(f"Actions called at stride ({action_stride})",
        step_actions >= expected_min_acts,
        f"{step_actions} >= {expected_min_acts}")
    chk("No NaN in final obs",              nan_count == 0)
    chk("No Inf in final obs",              inf_count == 0)
    chk("No errors in replay loop",         len(errors) == 0,
        f"{len(errors)} error(s)" if errors else "0")
    chk("Wrapper processed events",
        wrapper._events_processed > 0,
        f"events_processed={wrapper._events_processed}")
    chk("Agent acts count matches",
        agent_stats.get("n_acts", 0) == step_actions,
        f"agent={agent_stats.get('n_acts')} wrapper_calls={step_actions}")
    if n_trades > 0:
        chk("All trade PnLs are finite",
            all(math.isfinite(t.pnl_ticks) for t in trades))
        chk("Hold times >= 0",
            all(t.hold_secs >= 0 for t in trades))
        # Cost model spot-check (HC #231(A)): market exits cost 0.376 ticks RT (commission only)
        market_exits = [t for t in trades if t.exit_reason == "rl_close"]
        if market_exits:
            chk("Cost model correct (market exits)",
                all(
                    abs(t.pnl_ticks -
                        ((t.exit_price - t.entry_price) * t.direction / TICK_SIZE - 0.376)
                    ) < 0.02
                    for t in market_exits
                ),
                f"checked {len(market_exits)} market-exit trades at 0.376t cost"
            )
        else:
            chk("Cost model (no market exits to check)", True, "skipped — no rl_close trades")

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  RESULT: {n_pass} PASSED  {n_fail} FAILED  {len(warnings)} warnings")

    if n_fail > 0:
        print("  STATUS: FAILED — fix issues before Monday live deployment")
        print("=" * 72)
        return 1
    elif warnings:
        print("  STATUS: PASSED WITH WARNINGS — review before deployment")
        if not weights_loaded:
            print("  NOTE: Pipeline tested with RANDOM weights.")
            print("        Re-run with a matching checkpoint (OBS_DIM=48) for")
            print("        real P&L validation once training produces a new checkpoint.")
        print("=" * 72)
        return 2
    else:
        print("  STATUS: FULLY PASSED — pipeline is ready for Monday paper trading")
        print("=" * 72)
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end validation of the live RL trading pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with all defaults (200k events, CPU, random weights fallback):
  python3 test_rl_pipeline.py

  # Use specific checkpoint:
  python3 test_rl_pipeline.py --weights /path/to/checkpoint.pt

  # Replay more events with verbose action logging:
  python3 test_rl_pipeline.py --max-events 500000 --verbose

  # Override data directories:
  python3 test_rl_pipeline.py \\
      --data-dir /path/to/mbo_events \\
      --pred-dir /path/to/cnn_mamba_preds \\
      --patchtst-dir /path/to/patchtst_preds
""",
    )
    p.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help=f"Path to RL checkpoint .pt file (default: {DEFAULT_WEIGHTS})",
    )
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, dest="data_dir",
        help=f"Directory with MBO events NPZ files",
    )
    p.add_argument(
        "--pred-dir", type=Path, default=DEFAULT_PRED_DIR, dest="pred_dir",
        help=f"Directory with CNN-Mamba fold_XX_oot_predictions.npz",
    )
    p.add_argument(
        "--patchtst-dir", type=Path, default=DEFAULT_PST_DIR, dest="patchtst_dir",
        help=f"Directory with PatchTST fold_XX_oot_predictions.npz",
    )
    p.add_argument(
        "--max-events", type=int, default=200_000, dest="max_events",
        help="Max RTH events to replay (default: 200,000 ≈ a few minutes of RTH)",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=256, dest="hidden_dim",
        help="FIFOActorCritic hidden dim — must match checkpoint (default: 256)",
    )
    p.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="PyTorch device (default: cpu)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log every non-hold action during replay",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.verbose:
        for name in ("rl_execution_agent", "fifo_rl_env"):
            logging.getLogger(name).setLevel(logging.WARNING)

    sys.exit(run_pipeline_test(
        data_dir     = args.data_dir,
        pred_dir     = args.pred_dir,
        patchtst_dir = args.patchtst_dir,
        weights_path = args.weights,
        max_events   = args.max_events,
        hidden_dim   = args.hidden_dim,
        device       = args.device,
        verbose      = args.verbose,
    ))
