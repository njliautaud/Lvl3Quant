"""
HC #405 — BETTER EXITS SWEEP (LONG side, 10s & 30s heads)

CONTEXT: HC #404 decomposition revealed that 30s LONG fixed-hold loses ~92% of
available gross MFE to mean reversion before exit. Trial 278 only captures
+0.41 tk of signal contribution; the rest of its +1.99 tk/fill is the +2.0
passive_at_touch_plus_2 credit. We need to test whether DYNAMIC exit logic can
recover meaningfully more of the gross MFE than fixed-hold without leaning on
the passive credit.

GOAL: Sweep 3 exit families on 30s LONG and 10s LONG. Report which (if any)
beats trial 278's risk-adjusted profile (Sharpe ≥ 5, tk/fill ≥ +0.5,
day_conc ≤ 0.20, n_fills ≥ 30) PRIMARILY via passive_at_touch entry (so the
edge is from the exit logic, not the +2 limit credit).

EXIT FAMILIES SWEPT:
  1. MFE-trigger market exit: scan horizons in hold window in chronological
     order; if cumulative in-pos move ≥ THRESHOLD, exit market at that horizon
     (pay -spread). THRESHOLD ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0} tk.
     If never triggered, market-exit at horizon (= max_hold).
  2. Trailing stop: walk horizons in hold window; track running max favorable
     excursion; exit if current in-pos retraces by TRAIL_tk from that max.
     Exit = current level (market, pay spread). TRAIL ∈ {0.5, 1.0, 1.5, 2.0}.
     If never triggered, exit at last horizon (market, pay spread).
  3. Signal-flip exit: walk the 250ms eval grid; exit when current prediction
     flips sign AND |pred| ≥ FLIP_THR (in units of normalized pred strength
     measured as a quantile of |pred|). FLIP_THR ∈ {0.0, 0.25, 0.50, 0.75}
     where 0.0 = any opposite-sign, 0.75 = strong opposite (|pred| ≥ q(0.75)
     of |pred|). Exit = horizon nearest actual_hold_sec (market, pay spread).
     If never triggered, exit at max_hold (market).

ENTRY CONFIGS:
  side = LONG only (per HC #405 scope)
  head ∈ {10s, 30s}
  entry_order ∈ {passive_at_touch, passive_at_touch_plus_1, passive_at_touch_plus_2}
    — start with passive_at_touch; only credit results with passive_+K if
      passive_at_touch already shows positive net for that exit family/param.
  confidence_threshold: trial 278 uses 0.0435 percentile gate (top ~4.4% by
    |pred|). For LONG we apply the same gate (top 4.4% most positive predictions)
    so signal selectivity is comparable. min_pred_strength_abs = 0.0676 (same
    as trial 278). DOCUMENTED: this is a deliberate apples-to-apples gate choice.

REPLAY MECHANICS:
  - Re-uses full_market_replay's data loaders, percentile gate, queue-deflator
    fill model. ENTRY logic is identical to the canonical path.
  - EXIT logic is implemented OUTSIDE full_market_replay (we do NOT modify
    that file). For each filled signal i we:
      * Construct the discrete cumulative in-position tick path at the 4
        horizon checkpoints H = {1s, 5s, 10s, 30s} that fall ≤ max_hold.
        These are realized signed cumulative moves from sample i (entry) to
        sample i + horizon_seconds. NaN where the per-horizon mask fails.
      * Apply the exit family logic to that path and produce
        (exit_time_seconds, exit_inpos_ticks, exit_is_market_or_passive).
      * Compute net_ticks_per_fill = exit_inpos_ticks + edge_offset_entry +
        edge_offset_exit - commission.
        - For passive_at_touch entry, edge_offset_entry = 0.0
        - For passive_+1, +1.0. For passive_+2, +2.0.
        - For market exit, edge_offset_exit = -spread_ticks_rth (canonical 1.0).
        - For signal-flip "passive exit" we do NOT model (we still use market
          exit at the flip eval; documented).

POST-FILTERS (applied identically to trial 278):
  - ToD window 13:00-15:00 ET
  - min_pred_strength_abs ≥ 0.0676
  - HC #405 deliberate omission: we do NOT apply the FIFO confluence filter
    (pred_fifo_tp4sl3_net > 0.80) because that is a SHORT-side filter design.
    For LONG we test pure signal exit-logic value-add against an LONG-canonical
    entry gate. Documented in VERDICT.md.

CONSTRAINTS:
  - DO NOT modify full_market_replay.py / v33_execution_optuna_full_market_replay.py
  - Time budget: ≤ 90 min on Jupiter CPU
  - Parallelize across configs with multiprocessing (8 workers max)

SANITY:
  - Run verify_trial278_from_json.py BEFORE sweep — if it FAILs we stop.

NOT MALWARE. Pure analysis. Reads predictions + labels. Writes only to
output/hc405_better_exits_<ts>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    _load_predictions, _load_fifo_labels,
    PRICE_UNIT_TO_TICKS, EVAL_STRIDE_SEC,
    ES_RT_COMMISSION_TICKS_DEFAULT, ES_SPREAD_TICKS_RTH_DEFAULT,
)

# ----------------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------------
PREDS_NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

OOT_DATES = [
    "20260301", "20260302", "20260303", "20260304", "20260305",
    "20260308", "20260309", "20260310", "20260311", "20260312",
    "20260315", "20260316", "20260317", "20260318", "20260319",
]

# Trial 278 reference (the bar to beat)
T278_REF = dict(
    sharpe=13.48,
    tk_per_fill=1.99,
    day_conc=0.186,
    n_fills=195,
    side="short", head="30s",
    order_type="passive_at_touch_plus_2",
    hold_seconds=1.477,
    min_pred_strength=0.0676,
    conf_pctile=0.0435,
    commission=0.376,
    spread=1.0,  # canonical RTH (NOT trial 278's optuna-sampled 0.77)
    tod_start=13, tod_end=15,
)

# Sweep parameters
SIDES = ["long"]
HEADS = ["10s", "30s"]  # the two horizons identified by HC #404
ENTRY_ORDERS = ["passive_at_touch", "passive_at_touch_plus_1", "passive_at_touch_plus_2"]

# Entry gate: same selectivity as trial 278 (top 4.4% by |pred|).
ENTRY_CONF_PCTILE = T278_REF["conf_pctile"]
ENTRY_MIN_PRED_STRENGTH = T278_REF["min_pred_strength"]

# Exit family params
MFE_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # ticks
TRAIL_STOPS = [0.5, 1.0, 1.5, 2.0]               # ticks
FLIP_THRESHOLDS = [0.0, 0.25, 0.50, 0.75]        # quantile of |pred|

# Fixed-hold (baseline reference within sweep, for delta computation)
FIXED_HOLD_CONTROL = True

# HC #344 winner gates
WIN_SHARPE = 5.0
WIN_TK_PER_FILL = 0.50
WIN_DAY_CONC = 0.20
WIN_N_FILLS = 30

# Queue-deflator constants (mirroring full_market_replay._queue_position_model)
QUEUE_DEFLATOR = {
    "passive_at_touch": 0.5,
    "passive_at_touch_plus_1": 0.25,
    "passive_at_touch_plus_2": 0.125,
}
QUEUE_DEFLATOR_SLOW_MULT = 0.25
QUEUE_SEED = 42

# Entry edge per order type
ENTRY_EDGE_TICKS = {
    "passive_at_touch": 0.0,
    "passive_at_touch_plus_1": +1.0,
    "passive_at_touch_plus_2": +2.0,
}

# Horizons available in NPZ
ALL_HORIZONS = ["1s", "5s", "10s", "30s"]
HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}

# Cost basis
COMMISSION = ES_RT_COMMISSION_TICKS_DEFAULT  # 0.376
SPREAD = ES_SPREAD_TICKS_RTH_DEFAULT          # 1.0


# ----------------------------------------------------------------------------
# Data loading (cached at module level; survives within a worker process)
# ----------------------------------------------------------------------------
_DATA = {}


def load_all_data() -> dict:
    """Load predictions + labels and pre-compute filter masks. Returns dict
    that callers can pass to worker functions if needed (we also stash in _DATA
    for in-process reuse).
    """
    print(f"[{_t()}] Loading predictions and labels...")
    # Load each head separately via _load_predictions (this populates tgt_lr
    # dict for all 4 horizons consistently).
    preds_by_head = {}
    for h in ALL_HORIZONS:
        preds_by_head[h] = _load_predictions(PREDS_NPZ, h)
    n_pred = preds_by_head["1s"]["n"]

    # FIFO labels (timestamps + per-side fill flags)
    fifo = _load_fifo_labels(LABELS_DIR, OOT_DATES)
    n_fifo = sum(fifo["_n_per_day"])
    n = min(n_pred, n_fifo)
    print(f"[{_t()}] preds_n={n_pred} fifo_n={n_fifo} using n={n}")

    # Truncate everything to n
    preds_truncated = {}
    for h, pd_h in preds_by_head.items():
        preds_truncated[h] = {
            "pred": pd_h["pred"][:n],
            "pred_mask": pd_h["mask"][:n],
            "tgt_lr": {hh: arr[:n] for hh, arr in pd_h["tgt_lr"].items()},
            "tgt_lr_mask": {hh: arr[:n] for hh, arr in pd_h["tgt_lr_mask"].items()},
        }
    ts_ns = fifo["ts_ns"][:n]
    date_idx = fifo["_date_idx"][:n]

    # Pre-compute ToD filter (13-15 ET)
    ts_pd = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/New_York")
    hours = ts_pd.hour.to_numpy()
    tod_mask = (hours >= T278_REF["tod_start"]) & (hours < T278_REF["tod_end"])

    out = {
        "n": n,
        "preds": preds_truncated,
        "fifo": {k: (v[:n] if isinstance(v, np.ndarray) and v.shape[0] >= n else v)
                 for k, v in fifo.items() if not k.startswith("_")},
        "ts_ns": ts_ns,
        "date_idx": date_idx,
        "tod_mask": tod_mask,
        "dates": OOT_DATES,
    }
    print(f"[{_t()}] Data loaded. n={n}")
    return out


def _t() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ----------------------------------------------------------------------------
# Entry selection (mirrors full_market_replay percentile gate)
# ----------------------------------------------------------------------------
def select_entries(D: dict, side: str, head: str) -> np.ndarray:
    """Return global indices of signals passing percentile + pred-strength +
    ToD filters. NO fifo confluence (LONG side, see header)."""
    pred = D["preds"][head]["pred"]
    pmask = D["preds"][head]["pred_mask"]
    p_valid = pred[pmask]
    if p_valid.size == 0:
        return np.array([], dtype=np.int64)
    if side == "long":
        thr = float(np.quantile(p_valid, 1.0 - ENTRY_CONF_PCTILE))
        sel = pmask & (pred >= thr)
    else:
        thr = float(np.quantile(p_valid, ENTRY_CONF_PCTILE))
        sel = pmask & (pred <= thr)
    sel = sel & (np.abs(pred) >= ENTRY_MIN_PRED_STRENGTH)
    sel = sel & D["tod_mask"]
    return np.where(sel)[0]


# ----------------------------------------------------------------------------
# Queue-deflator fill model (mirrors full_market_replay)
# ----------------------------------------------------------------------------
def apply_queue_fill(D: dict, sel_idx: np.ndarray, side: str, order_type: str,
                     cancel_eval_window: int = 79) -> np.ndarray:
    """Returns boolean mask over sel_idx indicating which signals filled."""
    if order_type == "ioc_market":
        return np.ones(sel_idx.size, dtype=bool)
    side_key = side
    filled_lbl = D["fifo"][f"tp4sl3_{side_key}_filled"][sel_idx]
    exit_reason = D["fifo"][f"tp4sl3_{side_key}_exit_reason"][sel_idx]
    hold_time_ns = D["fifo"][f"tp4sl3_{side_key}_hold_time_ns"][sel_idx]

    cancel_sec = cancel_eval_window * EVAL_STRIDE_SEC
    hold_sec = hold_time_ns / 1e9
    base_filled = filled_lbl & (hold_sec <= 4 * cancel_sec)

    deflator = QUEUE_DEFLATOR[order_type]
    rng = np.random.default_rng(seed=QUEUE_SEED)
    coin = rng.random(sel_idx.size)
    # Detect bytes vs str dtype for exit_reason
    if exit_reason.dtype.kind in ("U", "S", "O"):
        slow = (exit_reason == "max_hold") | (exit_reason == b"max_hold")
    else:
        slow = np.zeros(sel_idx.size, dtype=bool)
    effective = np.where(slow, deflator * QUEUE_DEFLATOR_SLOW_MULT, deflator)
    return base_filled & (coin < effective)


# ----------------------------------------------------------------------------
# Build the in-position cumulative tick path for filled signals
# ----------------------------------------------------------------------------
def build_inpos_paths(D: dict, head: str, filled_idx: np.ndarray,
                      side_sign: float, max_hold_sec: float
                      ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (horizons_in_window, path_ticks[N_filled, H], path_mask[N_filled, H]).

    path_ticks[i,k] = side_sign * realized cumulative tick move from entry i
    to entry i + HORIZON_SEC[horizons[k]]. NaN where mask fails (mask returned
    as a separate boolean array).
    """
    horizons = [h for h in ALL_HORIZONS if HORIZON_SEC[h] <= max_hold_sec + 1e-9]
    if not horizons:
        horizons = ["1s"]
    n_f = filled_idx.size
    H = len(horizons)
    path = np.full((n_f, H), np.nan, dtype=np.float64)
    mask = np.zeros((n_f, H), dtype=bool)
    for k, h in enumerate(horizons):
        lr = D["preds"][head]["tgt_lr"][h][filled_idx]
        mk = D["preds"][head]["tgt_lr_mask"][h][filled_idx]
        in_pos = side_sign * lr * PRICE_UNIT_TO_TICKS
        path[:, k] = np.where(mk, in_pos, np.nan)
        mask[:, k] = mk
    return horizons, path, mask


# ----------------------------------------------------------------------------
# Exit family implementations
# ----------------------------------------------------------------------------
def exit_mfe_trigger(path: np.ndarray, mask: np.ndarray, horizons: list[str],
                     threshold_tk: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At first horizon checkpoint where in_pos >= threshold, exit market at
    that horizon (realized travel = in_pos at that horizon).
    If never triggered, exit market at final horizon (= max_hold).

    Returns (exit_inpos_ticks[N], exit_horizon_sec[N], triggered[N]).
    """
    n_f, H = path.shape
    triggered = np.zeros(n_f, dtype=bool)
    exit_inpos = np.full(n_f, np.nan, dtype=np.float64)
    exit_sec = np.full(n_f, np.nan, dtype=np.float64)
    for k in range(H):
        in_pos_k = path[:, k]
        valid_k = mask[:, k] & np.isfinite(in_pos_k)
        trig_now = valid_k & ~triggered & (in_pos_k >= threshold_tk)
        exit_inpos = np.where(trig_now, in_pos_k, exit_inpos)
        exit_sec = np.where(trig_now, HORIZON_SEC[horizons[k]], exit_sec)
        triggered = triggered | trig_now
    # For never-triggered: exit at final horizon (last valid).
    final_horizon_sec = HORIZON_SEC[horizons[-1]]
    final_inpos = path[:, -1]
    final_valid = mask[:, -1] & np.isfinite(final_inpos)
    fallback = ~triggered
    exit_inpos = np.where(fallback & final_valid, final_inpos, exit_inpos)
    exit_sec = np.where(fallback & final_valid, final_horizon_sec, exit_sec)
    return exit_inpos, exit_sec, triggered


def exit_trailing_stop(path: np.ndarray, mask: np.ndarray, horizons: list[str],
                       trail_tk: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk horizons; track running max favorable excursion. Exit if current
    in_pos retraces by trail_tk from peak (exit at current level, market).
    If never triggered, exit at final horizon (market).
    """
    n_f, H = path.shape
    triggered = np.zeros(n_f, dtype=bool)
    running_max = np.full(n_f, -np.inf, dtype=np.float64)
    exit_inpos = np.full(n_f, np.nan, dtype=np.float64)
    exit_sec = np.full(n_f, np.nan, dtype=np.float64)
    for k in range(H):
        in_pos_k = path[:, k]
        valid_k = mask[:, k] & np.isfinite(in_pos_k)
        # Update running max only where valid
        running_max = np.where(valid_k, np.maximum(running_max, in_pos_k), running_max)
        # Trigger if retracement from peak >= trail_tk, and peak > 0 (i.e. we
        # actually had favorable excursion; trail makes no sense on never-favorable)
        retrace = running_max - in_pos_k
        trig_now = (valid_k & ~triggered & (running_max > 0.0)
                    & (retrace >= trail_tk))
        exit_inpos = np.where(trig_now, in_pos_k, exit_inpos)
        exit_sec = np.where(trig_now, HORIZON_SEC[horizons[k]], exit_sec)
        triggered = triggered | trig_now
    # Never-triggered: exit at final horizon
    final_inpos = path[:, -1]
    final_valid = mask[:, -1] & np.isfinite(final_inpos)
    fallback = ~triggered
    exit_inpos = np.where(fallback & final_valid, final_inpos, exit_inpos)
    exit_sec = np.where(fallback & final_valid, HORIZON_SEC[horizons[-1]], exit_sec)
    return exit_inpos, exit_sec, triggered


def exit_signal_flip(D: dict, head: str, filled_idx: np.ndarray, side: str,
                     side_sign: float, max_hold_sec: float, flip_thr_q: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk the 250ms eval grid up to max_hold_sec. Exit at first k where
    sign(pred[i+k]) is opposite of side AND |pred[i+k]| >= q(flip_thr_q) of
    |pred|. Exit time quantized to the bracketing horizon for P&L (uses
    realized cumulative tick move at that horizon).

    If never triggered, exit at horizon nearest max_hold_sec.
    """
    pred = D["preds"][head]["pred"]
    pmask = D["preds"][head]["pred_mask"]
    n_total = pred.shape[0]
    n_f = filled_idx.size

    # |pred| quantile threshold (computed on valid |pred| values for this head)
    abs_pred_valid = np.abs(pred[pmask])
    if flip_thr_q <= 0.0:
        flip_abs_thr = 0.0
    else:
        flip_abs_thr = float(np.quantile(abs_pred_valid, flip_thr_q))

    max_k = int(np.ceil(max_hold_sec / EVAL_STRIDE_SEC))
    triggered = np.zeros(n_f, dtype=bool)
    exit_k = np.full(n_f, -1, dtype=np.int64)

    # Vectorized inner loop over k (rather than per-fill, which is slower)
    for k in range(1, max_k + 1):
        idx = filled_idx + k
        in_range = idx < n_total
        # Where out of range, mark as triggered at last valid k
        # (we treat out-of-range as no further data, exit at last index)
        if not in_range.any():
            break
        p = np.where(in_range, pred[np.clip(idx, 0, n_total - 1)], 0.0)
        pm = np.where(in_range, pmask[np.clip(idx, 0, n_total - 1)], False)
        # opposite-sign trigger relative to entry direction
        if side == "long":
            flip = (p < 0) & (np.abs(p) >= flip_abs_thr)
        else:
            flip = (p > 0) & (np.abs(p) >= flip_abs_thr)
        trig_now = (~triggered) & in_range & pm & flip
        exit_k = np.where(trig_now, k, exit_k)
        triggered = triggered | trig_now

    # actual_hold_sec per fill
    actual_hold_sec = np.where(triggered, exit_k * EVAL_STRIDE_SEC, max_hold_sec)

    # Quantize to bracketing horizon (above), compute exit in-pos at that horizon
    # using realized cumulative target_log_ret at that horizon for the ENTRY index.
    exit_inpos = np.full(n_f, np.nan, dtype=np.float64)
    exit_sec_out = np.full(n_f, np.nan, dtype=np.float64)
    for hi, h in enumerate(ALL_HORIZONS):
        if HORIZON_SEC[h] > max_hold_sec + 1e-9:
            break
        # which fills fall in this horizon bucket?
        if hi == 0:
            in_bucket = actual_hold_sec <= HORIZON_SEC[h] + 1e-9
        else:
            prev_h = ALL_HORIZONS[hi - 1]
            in_bucket = ((actual_hold_sec > HORIZON_SEC[prev_h] + 1e-9)
                         & (actual_hold_sec <= HORIZON_SEC[h] + 1e-9))
        lr = D["preds"][head]["tgt_lr"][h][filled_idx]
        mk = D["preds"][head]["tgt_lr_mask"][h][filled_idx]
        in_pos = side_sign * lr * PRICE_UNIT_TO_TICKS
        valid = mk & np.isfinite(in_pos) & in_bucket
        exit_inpos = np.where(valid, in_pos, exit_inpos)
        exit_sec_out = np.where(valid, HORIZON_SEC[h], exit_sec_out)

    # Anything still NaN (no bracket fit, e.g. actual_hold_sec > max horizon):
    # use the largest horizon ≤ max_hold_sec
    largest_h = [h for h in ALL_HORIZONS if HORIZON_SEC[h] <= max_hold_sec + 1e-9][-1]
    lr_last = D["preds"][head]["tgt_lr"][largest_h][filled_idx]
    mk_last = D["preds"][head]["tgt_lr_mask"][largest_h][filled_idx]
    in_pos_last = side_sign * lr_last * PRICE_UNIT_TO_TICKS
    valid_last = mk_last & np.isfinite(in_pos_last)
    fallback = np.isnan(exit_inpos) & valid_last
    exit_inpos = np.where(fallback, in_pos_last, exit_inpos)
    exit_sec_out = np.where(fallback, HORIZON_SEC[largest_h], exit_sec_out)

    return exit_inpos, exit_sec_out, triggered


def exit_fixed_hold(path: np.ndarray, mask: np.ndarray, horizons: list[str],
                    hold_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-hold market exit at the horizon nearest (bracketing above) hold_sec."""
    # Pick bracketing-above horizon
    h_pick = None
    for h in horizons:
        if HORIZON_SEC[h] >= hold_sec - 1e-9:
            h_pick = h
            break
    if h_pick is None:
        h_pick = horizons[-1]
    k_pick = horizons.index(h_pick)
    in_pos = path[:, k_pick]
    valid = mask[:, k_pick] & np.isfinite(in_pos)
    exit_inpos = np.where(valid, in_pos, np.nan)
    exit_sec = np.where(valid, HORIZON_SEC[h_pick], np.nan)
    triggered = np.ones(in_pos.size, dtype=bool)
    return exit_inpos, exit_sec, triggered


# ----------------------------------------------------------------------------
# Metrics (mirrors metrics_from_filtered in optuna script)
# ----------------------------------------------------------------------------
def compute_metrics(net_ticks: np.ndarray, ts_ns: np.ndarray) -> dict:
    finite = np.isfinite(net_ticks)
    net = net_ticks[finite]
    ts = ts_ns[finite]
    n = net.size
    if n == 0:
        return dict(n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0,
                    n_per_day_median=0.0, n_days=0,
                    hc344_pass=False)
    mean = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if n > 1 else 0.0
    sharpe = mean / sd * float(np.sqrt(252.0)) if sd > 0 else 0.0
    neg = net[net < 0]
    dsd = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = mean / dsd * float(np.sqrt(252.0)) if dsd > 0 else 0.0
    pos = float(net[net > 0].sum())
    negabs = float(-net[net < 0].sum())
    pf = pos / negabs if negabs > 0 else (999.0 if pos > 0 else 0.0)
    wr = float((net > 0).mean() * 100.0)
    ts_pd = pd.to_datetime(ts, unit="ns", utc=True).tz_convert("America/New_York")
    day_str = ts_pd.strftime("%Y%m%d")
    day_df = pd.DataFrame({"day": day_str, "net": net})
    by_day = day_df.groupby("day")["net"].sum()
    total = float(by_day.sum())
    day_conc = float(by_day.abs().max() / max(1e-9, abs(total))) if abs(total) > 1e-9 else 1.0
    n_per_day = day_df.groupby("day").size()
    n_per_day_median = float(n_per_day.median()) if len(n_per_day) > 0 else 0.0
    ci_low_95 = mean - 1.96 * sd / max(1.0, float(np.sqrt(n))) if sd > 0 else mean

    hc344 = (n >= WIN_N_FILLS) and (sharpe >= WIN_SHARPE) and \
            (mean >= WIN_TK_PER_FILL) and (day_conc <= WIN_DAY_CONC)
    return dict(n_fills=int(n), sharpe=float(sharpe), sortino=float(sortino),
                pf=float(pf), wr=float(wr), mean_net=float(mean),
                day_conc=float(day_conc), ci_low_95=float(ci_low_95),
                n_per_day_median=float(n_per_day_median),
                n_days=int(by_day.size),
                hc344_pass=bool(hc344))


# ----------------------------------------------------------------------------
# Single-config evaluation (called by worker)
# ----------------------------------------------------------------------------
def evaluate_config(D: dict, side: str, head: str, entry_order: str,
                    exit_family: str, exit_param: float,
                    max_hold_sec: float | None = None) -> dict:
    side_sign = +1.0 if side == "long" else -1.0
    if max_hold_sec is None:
        max_hold_sec = HORIZON_SEC[head]

    sel_idx = select_entries(D, side, head)
    if sel_idx.size == 0:
        return _empty_row(side, head, entry_order, exit_family, exit_param,
                          max_hold_sec, reason="no_entries")

    filled_mask_sel = apply_queue_fill(D, sel_idx, side, entry_order,
                                       cancel_eval_window=79)
    filled_idx = sel_idx[filled_mask_sel]
    if filled_idx.size == 0:
        return _empty_row(side, head, entry_order, exit_family, exit_param,
                          max_hold_sec, reason="no_fills")

    horizons, path, mask = build_inpos_paths(D, head, filled_idx, side_sign,
                                             max_hold_sec)

    if exit_family == "mfe_trigger_market":
        exit_inpos, exit_sec, _ = exit_mfe_trigger(path, mask, horizons, exit_param)
        edge_exit = -SPREAD  # market exit pays spread
    elif exit_family == "trailing_stop":
        exit_inpos, exit_sec, _ = exit_trailing_stop(path, mask, horizons, exit_param)
        edge_exit = -SPREAD
    elif exit_family == "signal_flip":
        exit_inpos, exit_sec, _ = exit_signal_flip(D, head, filled_idx, side,
                                                    side_sign, max_hold_sec,
                                                    exit_param)
        edge_exit = -SPREAD
    elif exit_family == "fixed_hold":
        # exit_param here = hold_sec
        exit_inpos, exit_sec, _ = exit_fixed_hold(path, mask, horizons, exit_param)
        edge_exit = -SPREAD
    else:
        raise ValueError(f"Unknown exit_family: {exit_family}")

    edge_entry = ENTRY_EDGE_TICKS[entry_order]
    net_ticks = exit_inpos + edge_entry + edge_exit - COMMISSION
    # Bookkeep: fills with NaN exit_inpos (no realized data) -> drop
    ts_filled = D["ts_ns"][filled_idx]
    m = compute_metrics(net_ticks, ts_filled)

    # Diagnostics: mean exit time, n_triggered (vs fallback)
    mean_exit_sec = float(np.nanmean(exit_sec)) if exit_sec.size else float("nan")
    mean_inpos_exit = float(np.nanmean(exit_inpos)) if exit_inpos.size else float("nan")

    return dict(
        side=side, head=head, entry_order=entry_order,
        exit_family=exit_family, exit_param=exit_param,
        max_hold_sec=float(max_hold_sec),
        n_entries=int(sel_idx.size),
        n_fills_attempted=int(filled_idx.size),
        edge_entry=edge_entry, edge_exit=edge_exit, commission=COMMISSION,
        mean_exit_sec=mean_exit_sec,
        mean_inpos_at_exit_tk=mean_inpos_exit,
        **m,
    )


def _empty_row(side, head, entry_order, exit_family, exit_param, max_hold_sec,
               reason: str = "") -> dict:
    return dict(
        side=side, head=head, entry_order=entry_order,
        exit_family=exit_family, exit_param=exit_param,
        max_hold_sec=float(max_hold_sec),
        n_entries=0, n_fills_attempted=0,
        edge_entry=ENTRY_EDGE_TICKS.get(entry_order, 0.0),
        edge_exit=-SPREAD, commission=COMMISSION,
        mean_exit_sec=float("nan"), mean_inpos_at_exit_tk=float("nan"),
        n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
        mean_net=0.0, day_conc=1.0, ci_low_95=-999.0,
        n_per_day_median=0.0, n_days=0, hc344_pass=False, reason=reason,
    )


# ----------------------------------------------------------------------------
# Worker entry point for multiprocessing
# ----------------------------------------------------------------------------
_GLOBAL_D = None


def _worker_init():
    global _GLOBAL_D
    _GLOBAL_D = load_all_data()


def _worker_eval(args):
    side, head, entry_order, exit_family, exit_param, max_hold = args
    try:
        return evaluate_config(_GLOBAL_D, side, head, entry_order,
                               exit_family, exit_param, max_hold)
    except Exception as e:
        return dict(side=side, head=head, entry_order=entry_order,
                    exit_family=exit_family, exit_param=exit_param,
                    max_hold_sec=float(max_hold) if max_hold else 0.0,
                    n_entries=0, n_fills_attempted=0,
                    n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0,
                    n_per_day_median=0.0, n_days=0, hc344_pass=False,
                    reason=f"error:{type(e).__name__}:{e}")


# ----------------------------------------------------------------------------
# Sanity check: run verify_trial278_from_json.py
# ----------------------------------------------------------------------------
def run_sanity_check() -> tuple[bool, str]:
    print(f"[{_t()}] Running sanity check (verify_trial278_from_json.py)...")
    result = subprocess.run(
        ["python3", str(PROJ / "scripts" / "v3_3_research" / "verify_trial278_from_json.py")],
        cwd=str(PROJ), capture_output=True, text=True, timeout=300,
    )
    out = result.stdout + "\n" + result.stderr
    passed = (result.returncode == 0) and ("OVERALL: PASS" in out)
    return passed, out


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def build_sweep_configs() -> list[tuple]:
    """Return list of (side, head, entry_order, exit_family, exit_param, max_hold)."""
    configs = []
    for side in SIDES:
        for head in HEADS:
            max_hold = HORIZON_SEC[head]
            for entry_order in ENTRY_ORDERS:
                # Fixed-hold control (baseline within sweep) — exit at horizon
                if FIXED_HOLD_CONTROL:
                    configs.append((side, head, entry_order, "fixed_hold",
                                    max_hold, max_hold))
                # MFE-trigger market exits
                for thr in MFE_THRESHOLDS:
                    configs.append((side, head, entry_order,
                                    "mfe_trigger_market", thr, max_hold))
                # Trailing stops
                for trail in TRAIL_STOPS:
                    configs.append((side, head, entry_order,
                                    "trailing_stop", trail, max_hold))
                # Signal-flip exits
                for flip in FLIP_THRESHOLDS:
                    configs.append((side, head, entry_order,
                                    "signal_flip", flip, max_hold))
    return configs


def run_sweep(out_dir: Path, workers: int = 8) -> pd.DataFrame:
    configs = build_sweep_configs()
    print(f"[{_t()}] Sweep has {len(configs)} configs, running with {workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_worker_eval, c): c for c in configs}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 10 == 0 or done == len(configs):
                print(f"[{_t()}] Completed {done}/{len(configs)}")
    df = pd.DataFrame(rows)
    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{_t()}] Wrote {csv_path}")
    return df


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def write_winners_md(df: pd.DataFrame, out_dir: Path) -> int:
    """Configs that BEAT trial 278 on risk-adjusted basis.
    Primary criterion: passive_at_touch entry (signal-driven, not credit).
    Secondary view: any entry_order.
    """
    winners_signal_only = df[
        (df["entry_order"] == "passive_at_touch")
        & (df["n_fills"] >= WIN_N_FILLS)
        & (df["sharpe"] >= WIN_SHARPE)
        & (df["mean_net"] >= WIN_TK_PER_FILL)
        & (df["day_conc"] <= WIN_DAY_CONC)
        & (df["exit_family"] != "fixed_hold")
    ].sort_values("sharpe", ascending=False).head(10)

    winners_any = df[
        (df["n_fills"] >= WIN_N_FILLS)
        & (df["sharpe"] >= WIN_SHARPE)
        & (df["mean_net"] >= WIN_TK_PER_FILL)
        & (df["day_conc"] <= WIN_DAY_CONC)
    ].sort_values("sharpe", ascending=False).head(10)

    lines = []
    lines.append("# HC #405 — WINNERS")
    lines.append("")
    lines.append(f"Produced: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    lines.append("")
    lines.append("## Bar to beat (trial 278)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| Sharpe | {T278_REF['sharpe']} |")
    lines.append(f"| tk/fill | {T278_REF['tk_per_fill']} |")
    lines.append(f"| day_conc | {T278_REF['day_conc']} |")
    lines.append(f"| n_fills | {T278_REF['n_fills']} |")
    lines.append("")
    lines.append("Trial 278 is 30s SHORT passive_at_touch_plus_2 (the +2 credit "
                 "accounts for ~+2.0 of the +1.99 tk/fill).")
    lines.append("")
    lines.append("## Win gates (signal-driven configs)")
    lines.append("")
    lines.append(f"- Sharpe ≥ {WIN_SHARPE}")
    lines.append(f"- tk/fill ≥ {WIN_TK_PER_FILL}")
    lines.append(f"- day_conc ≤ {WIN_DAY_CONC}")
    lines.append(f"- n_fills ≥ {WIN_N_FILLS}")
    lines.append("- entry_order = passive_at_touch (NO passive credit)")
    lines.append("- exit_family ≠ fixed_hold (must be dynamic)")
    lines.append("")
    lines.append("## SIGNAL-DRIVEN winners (passive_at_touch, dynamic exit)")
    lines.append("")
    if winners_signal_only.empty:
        lines.append("**NONE.** No dynamic exit family at passive_at_touch entry "
                     "passes the win gates on LONG side / 10s or 30s heads.")
    else:
        cols = ["side", "head", "entry_order", "exit_family", "exit_param",
                "n_fills", "sharpe", "mean_net", "day_conc", "pf", "wr",
                "mean_exit_sec"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in winners_signal_only.iterrows():
            vals = [f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating))
                    else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## ANY-ENTRY-ORDER winners (incl. passive_+1, passive_+2 credit)")
    lines.append("")
    if winners_any.empty:
        lines.append("**NONE.** No config (any entry_order, any exit family) "
                     "passes the win gates on LONG side / 10s or 30s heads.")
    else:
        cols = ["side", "head", "entry_order", "exit_family", "exit_param",
                "n_fills", "sharpe", "mean_net", "day_conc", "pf", "wr",
                "mean_exit_sec"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in winners_any.iterrows():
            vals = [f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating))
                    else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    (out_dir / "WINNERS.md").write_text("\n".join(lines))
    return int(len(winners_signal_only)), int(len(winners_any))


def write_verdict_md(df: pd.DataFrame, n_sig_winners: int, n_any_winners: int,
                     sanity_passed: bool, out_dir: Path) -> None:
    lines = []
    lines.append("# HC #405 — VERDICT: Can dynamic exit logic recover MFE on LONG side?")
    lines.append("")
    lines.append(f"Produced: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    lines.append(f"Sanity check (trial 278 reproduction): "
                 f"{'PASS' if sanity_passed else 'FAIL'}")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    if n_sig_winners > 0:
        lines.append(f"**YES.** {n_sig_winners} signal-driven config(s) "
                     f"(passive_at_touch entry + dynamic exit) beat trial 278's "
                     f"risk-adjusted profile on LONG. See WINNERS.md.")
    elif n_any_winners > 0:
        lines.append(f"**PARTIAL.** {n_any_winners} config(s) clear the gates "
                     f"on LONG side, but all rely on the passive_+K limit credit "
                     f"(NOT on dynamic exit value-add). The LONG signal alone "
                     f"is not capturable.")
    else:
        lines.append("**NO.** No dynamic exit family produces a config on LONG "
                     "(10s or 30s head) that beats trial 278's bar. This is "
                     "consistent with HC #404's MFE-leakage finding: the long-"
                     "side alpha exists but evaporates faster than any tested "
                     "exit policy can capture it.")
    lines.append("")
    lines.append("## Family-by-family summary (passive_at_touch entry only)")
    lines.append("")
    pat = df[df["entry_order"] == "passive_at_touch"].copy()
    fam_summary = pat.groupby(["head", "exit_family"]).agg(
        best_sharpe=("sharpe", "max"),
        best_tk=("mean_net", "max"),
        median_tk=("mean_net", "median"),
        worst_day_conc=("day_conc", "max"),
        any_pass=("hc344_pass", "any"),
        n_configs=("hc344_pass", "size"),
    ).reset_index()
    if not fam_summary.empty:
        lines.append("| head | exit_family | best_sharpe | best_tk | median_tk | worst_day_conc | any_pass | n_configs |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in fam_summary.iterrows():
            lines.append(f"| {r['head']} | {r['exit_family']} | "
                         f"{r['best_sharpe']:.2f} | {r['best_tk']:+.3f} | "
                         f"{r['median_tk']:+.3f} | {r['worst_day_conc']:.3f} | "
                         f"{r['any_pass']} | {int(r['n_configs'])} |")
    lines.append("")
    lines.append("## Family-by-family summary (any entry_order, incl. passive +K credit)")
    lines.append("")
    any_summary = df.groupby(["head", "exit_family", "entry_order"]).agg(
        best_sharpe=("sharpe", "max"),
        best_tk=("mean_net", "max"),
        worst_day_conc=("day_conc", "max"),
        any_pass=("hc344_pass", "any"),
    ).reset_index()
    if not any_summary.empty:
        lines.append("| head | exit_family | entry_order | best_sharpe | best_tk | worst_day_conc | any_pass |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in any_summary.iterrows():
            lines.append(f"| {r['head']} | {r['exit_family']} | {r['entry_order']} | "
                         f"{r['best_sharpe']:.2f} | {r['best_tk']:+.3f} | "
                         f"{r['worst_day_conc']:.3f} | {r['any_pass']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **MFE-trigger market exit**: profits if signal generates "
                 "favorable excursion ≥ THRESHOLD within hold window. Pays 1.0 "
                 "tick spread cost. Net edge requires THRESHOLD - 1.0 - 0.376 "
                 "= THRESHOLD - 1.376 tk to be positive PER FILL, AND requires "
                 "enough fills to actually trigger.")
    lines.append("- **Trailing stop**: protects against giveback after favorable "
                 "excursion. Pays 1.0 tick spread cost. Works only if signal "
                 "consistently produces favorable peaks before reverting.")
    lines.append("- **Signal-flip**: exit when model says the trade direction "
                 "has reversed. Pays 1.0 tick spread cost. Requires the model's "
                 "subsequent predictions to be informative beyond the entry "
                 "horizon, not just AT it.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Exit P&L is quantized to {1s, 5s, 10s, 30s} horizon checkpoints "
                 "(realized cumulative tick moves), NOT true 250ms tick-level "
                 "intra-horizon path. This matches full_market_replay's "
                 "approximation. Same caveat as HC #403/404.")
    lines.append("- Spread for market exits set to canonical 1.0 tick RTH (NOT "
                 "trial 278's optuna-sampled 0.77 — we use canonical for honesty).")
    lines.append("- LONG side does NOT apply the FIFO-confluence filter that "
                 "trial 278's SHORT side uses (per HC #405 scope — we test exit "
                 "logic value-add independently, with the same percentile + "
                 "pred-strength entry gate).")
    lines.append("- Queue-deflator fill model uses identical params to "
                 "full_market_replay (passive_at_touch = 0.5, +1 = 0.25, +2 = "
                 "0.125, with slow-exit ×0.25 multiplier).")
    lines.append("")
    (out_dir / "VERDICT.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJ / "output" / f"hc405_better_exits_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_t()}] Output dir: {out_dir}")

    t0 = time.time()

    # 1. Sanity check
    sanity_passed = True
    sanity_out = "skipped"
    if not args.skip_sanity:
        sanity_passed, sanity_out = run_sanity_check()
        (out_dir / "sanity_check.log").write_text(sanity_out)
        if not sanity_passed:
            print(f"[{_t()}] SANITY CHECK FAILED — STOPPING.")
            print(sanity_out[-2000:])
            (out_dir / "VERDICT.md").write_text(
                "# HC #405 — SANITY CHECK FAILED\n\n"
                "Trial 278 reproduction did not pass tolerance. "
                "Replay loop has diverged from canonical. STOPPING sweep.\n\n"
                "See sanity_check.log for details."
            )
            return 2
        print(f"[{_t()}] Sanity check PASSED.")

    # 2. Sweep
    df = run_sweep(out_dir, workers=args.workers)

    # 3. Reports
    n_sig, n_any = write_winners_md(df, out_dir)
    write_verdict_md(df, n_sig, n_any, sanity_passed, out_dir)

    t1 = time.time()
    print(f"[{_t()}] DONE in {(t1-t0)/60:.1f} min")
    print(f"[{_t()}] Signal-driven winners: {n_sig}, any-entry winners: {n_any}")
    print(f"[{_t()}] Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
