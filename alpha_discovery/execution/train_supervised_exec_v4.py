#!/usr/bin/env python3
"""
Supervised Execution Predictor v4 — Full-Feature MLP for Trade Outcome Prediction
==================================================================================

Per HC #253, #254 (DIRECTIVES.md, 2026-05-08).

Extends v3 (`train_supervised_exec.py`) with the FULL feature vector the user
asked for: PatchTST predictions + agreement, vol regime bucket, queue positions,
time-of-day buckets, recent fill rate (placeholder), in addition to v3's
CNN-Mamba + microstructure features.

Per HC #254: A "positive" sweep result that conditions on only 2 axes is GARBAGE.
The right approach is a learned model on the full feature vector. That's this.

================================================================================
DATA SOURCES (read-only — NEVER mutate Neptune files)
================================================================================
- MBO events:       /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/
- CNN-Mamba preds:  /home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot/{date}_predictions.npz
- PatchTST preds:   /home/jupiter/Lvl3Quant/output/patchtst_bulk_oot/{date}_predictions.npz
- Fill sim labels:  /home/nick/Lvl3Quant/output/wide_tp_queue_sweep/{cfg}_{date}.json
                    (mirrored to /home/jupiter/Lvl3Quant/data/fill_sim_cache/wide_tp_queue_sweep/)

================================================================================
FEATURE VECTOR (printed at startup — see FEATURE_NAMES_V4 below)
================================================================================
v3 carryovers (19): pred_1s, pred_5s, pred_10s, abs versions, confidence_tier,
  book_imbalance, bid_depth_log, ask_depth_log, spread, recent_volatility,
  time_of_day (cont.), volume_imbalance, signal_agreement, signal_strength,
  price_momentum, pred_std, pred_range.

v4 additions (11):
  20-22 patchtst_pred_{1s,5s,10s}    -- PatchTST signal model
  23    patchtst_abs_pred_1s
  24    pt_mamba_sign_agree           -- 1 if signs match, else 0
  25    pt_mamba_mag_ratio            -- |patchtst| / |mamba|, clipped 0..5
  26    vol_regime_bucket             -- {0,1,2} from rolling-vol percentile within day
  27    tod_bucket                    -- {0,1,2} = open / mid / close
  28-29 queue_pos_bid, queue_pos_ask  -- from MBO event cols 11, 12 (precomputed)
  30    recent_fill_rate              -- PLACEHOLDER 0.0 (TODO HC #253(f))

LABEL: pnl_ticks per trade from FIFO fill_sim (commission already included = 0.376t).
       NO spread crossing cost (HC #231).

================================================================================
WALK-FORWARD: 5-fold sliding (HC #0). NEVER expanding.
================================================================================

MLflow: experiment "supervised_exec_v4". Tracking URI from env or default jupiter:5000.

Author: research code per user spec. Date: 2026-05-08.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse v3 helpers — do NOT rewrite the wheel.
sys.path.insert(0, str(Path(__file__).parent))
from train_supervised_exec import (  # noqa: E402
    COL_PRICE_REL, COL_QTY_LOG, COL_SIDE, COL_SPREAD,
    COMMISSION_TICKS, ES_TICK_VALUE, PRED_STRIDE, PRED_WINDOW,
    _time_of_day_fraction,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
DEFAULT_DATA_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
DEFAULT_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot"
DEFAULT_PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_bulk_oot"
DEFAULT_FILLSIM_DIR = LVL3_ROOT / "data" / "fill_sim_cache" / "wide_tp_queue_sweep"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "supervised_exec_v4"
DEFAULT_LOG_PATH = LVL3_ROOT / "logs" / "supervised_exec_v4_build.log"

DEFAULT_FILLSIM_CONFIG = "tp6_sl3_h60000_q3_t0.5"  # wider/looser → more labels per day

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
EXPERIMENT_NAME = "supervised_exec_v4"  # legacy default; per-head names below
EXPERIMENT_NAME_CONFLUENCE = "supervised_exec_v4_confluence"
EXPERIMENT_NAME_INVERSE = "supervised_exec_v4_inverse"

# Head-mode constants per HC #255 (PatchTST direction / CNN-Mamba magnitude inverse test)
HEAD_CONFLUENCE = "confluence"
HEAD_INVERSE = "inverse"
INVERSE_MODE_PREGATE = "pregate"   # only train on rows where sign(patchtst_1s) == sign(realized_dir)
INVERSE_MODE_FEATURE = "feature"   # train on all rows; add patchtst_sign_correct as binary feature

# ---------------------------------------------------------------------------
# Feature names — v4 vector (28 features)
# ---------------------------------------------------------------------------
FEATURE_NAMES_V4 = [
    # v3 carryover (19)
    "pred_1s", "pred_5s", "pred_10s",
    "abs_pred_1s", "abs_pred_5s", "abs_pred_10s",
    "confidence_tier",
    "book_imbalance", "bid_depth_log", "ask_depth_log", "spread",
    "recent_volatility", "time_of_day",
    "volume_imbalance", "signal_agreement", "signal_strength",
    "price_momentum", "pred_std", "pred_range",
    # v4 additions (11)
    "patchtst_pred_1s", "patchtst_pred_5s", "patchtst_pred_10s",
    "patchtst_abs_pred_1s",
    "pt_mamba_sign_agree", "pt_mamba_mag_ratio",
    "vol_regime_bucket", "tod_bucket",
    "queue_pos_bid", "queue_pos_ask",
    "recent_fill_rate",
]
N_FEATURES_V4 = len(FEATURE_NAMES_V4)  # 30

# Label is single per-trade pnl_ticks
LABEL_NAME = "pnl_ticks"

# ---------------------------------------------------------------------------
# Logging — to file AND console (HC #0 mandatory log)
# ---------------------------------------------------------------------------
DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [SUP_EXEC_V4] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DEFAULT_LOG_PATH), mode="a"),
    ],
)
log = logging.getLogger("supervised_exec_v4")


# ===========================================================================
# Label loader — fill_sim trade records → per-prediction-index labels
# ===========================================================================

def load_fill_sim_labels(
    fillsim_path: Path,
) -> Dict[int, float]:
    """
    Load fill_sim json, return dict mapping signal_time_ns -> pnl_ticks.

    The fill_sim json has trades[*] = { signal_time_ns, pnl_ticks, side, ... }.
    We use signal_time_ns as the key because pred_idx isn't stored in the trade
    record. Caller is responsible for mapping signal_time_ns -> pred_idx via the
    MBO event timestamps + PRED_STRIDE.

    Returns empty dict if file missing or no trades.
    """
    if not fillsim_path.exists():
        return {}
    try:
        with open(fillsim_path) as f:
            d = json.load(f)
    except Exception as e:
        log.warning(f"Failed to read {fillsim_path}: {e}")
        return {}
    out: Dict[int, float] = {}
    for tr in d.get("trades", []):
        ts_ns = int(tr.get("signal_time_ns", 0))
        pnl = float(tr.get("pnl_ticks", 0.0))
        if ts_ns > 0:
            out[ts_ns] = pnl
    return out


def map_signal_ts_to_pred_idx(
    timestamps_ns: np.ndarray,
    pred_stride: int = PRED_STRIDE,
    pred_window: int = PRED_WINDOW,
    n_preds: int = -1,
) -> Dict[int, int]:
    """
    Build inverse map: signal_time_ns -> pred_idx.
    pred_idx P corresponds to event index pred_window + P * pred_stride.
    timestamps_ns is the per-event timestamp array from the MBO npz.
    """
    if n_preds < 0:
        n_preds = (len(timestamps_ns) - pred_window) // pred_stride
    out: Dict[int, int] = {}
    for p in range(n_preds):
        ev_idx = pred_window + p * pred_stride
        if ev_idx < len(timestamps_ns):
            out[int(timestamps_ns[ev_idx])] = p
    return out


# ===========================================================================
# Feature extraction (v4) — extends v3 with PatchTST + bucketed regimes + queue
# ===========================================================================

def extract_v4_for_date(
    mbo_path: Path,
    mamba_pred_path: Path,
    patchtst_pred_path: Path,
    fillsim_path: Path,
    head: str = HEAD_CONFLUENCE,
    inverse_mode: str = INVERSE_MODE_FEATURE,
) -> Dict[str, np.ndarray]:
    """
    For one OOT date:
      - load MBO events
      - load CNN-Mamba & PatchTST predictions, align rows to min length
      - build v4 feature vector per prediction event
      - load fill_sim trade labels, attach pnl_ticks to predictions that became trades

    Head modes (per HC #255):
      head=confluence:
        Standard v4 head — CNN-Mamba sign decides direction at prediction time;
        PatchTST sign-agreement is just a filter feature already in the vector
        (`pt_mamba_sign_agree`). Target = pnl_ticks. No pre-gate.
      head=inverse:
        PatchTST sign decides direction; CNN-Mamba pred_{1s,5s,10s} provide
        magnitude/edge features. Two sub-modes:
          inverse_mode=pregate: drop rows where sign(patchtst_1s) != sign(realized_dir).
            "realized_dir" here = sign(pnl_ticks) for the recorded trade — i.e. we
            keep only the trades where PatchTST got direction right. Target =
            pnl_ticks. Model learns: GIVEN patchtst was right, when is it profitable?
          inverse_mode=feature: keep all rows; APPEND a binary
            `patchtst_sign_correct` feature (1 if sign(patchtst_1s) == sign(pnl_ticks)
            else 0). Model learns the gate itself. Target = pnl_ticks.

    Returns dict with:
      features (n_labeled, n_feats) float32   -- n_feats = N_FEATURES_V4 (or +1 in inverse-feature mode)
      labels   (n_labeled,)         float32   -- pnl_ticks
      meta     (n_labeled, 3)       float32   -- [ts_s, pred_idx, signal_dir]
      feature_names (list[str])
      n_predictions_total (int)
      n_labeled (int)
      date_str (str)
      head (str), inverse_mode (str)
    """
    if head not in (HEAD_CONFLUENCE, HEAD_INVERSE):
        raise ValueError(f"head must be {HEAD_CONFLUENCE!r} or {HEAD_INVERSE!r}, got {head!r}")
    if inverse_mode not in (INVERSE_MODE_PREGATE, INVERSE_MODE_FEATURE):
        raise ValueError(
            f"inverse_mode must be {INVERSE_MODE_PREGATE!r} or {INVERSE_MODE_FEATURE!r}, got {inverse_mode!r}"
        )
    date_str = mbo_path.stem.replace("_mbo_events", "")

    mbo = np.load(str(mbo_path))
    events = mbo["events"]              # (N, F) float32
    timestamps_ns = mbo["timestamps"]   # (N,) int64

    n_events = len(events)
    # We'll re-validate once we know actual window/stride below.

    # Predictions
    mamba = np.load(str(mamba_pred_path), allow_pickle=True)
    pt = np.load(str(patchtst_pred_path), allow_pickle=True)
    mp = mamba["predictions"]   # (Nm, 3)
    ml = mamba["labels"]        # (Nm, 3)
    pp = pt["predictions"]      # (Np, 3)

    # Read true window/stride from the npz files — v3 hardcoded 1000/50 but the
    # CNN-Mamba bulk_oot files actually use window=3000 stride=250 (and PatchTST
    # uses window=500 stride=250). We trust the file metadata.
    mamba_window = int(mamba["window_size"]) if "window_size" in mamba.files else PRED_WINDOW
    mamba_stride = int(mamba["stride"]) if "stride" in mamba.files else PRED_STRIDE
    pt_window = int(pt["window_size"]) if "window_size" in pt.files else PRED_WINDOW
    pt_stride = int(pt["stride"]) if "stride" in pt.files else PRED_STRIDE

    if mamba_stride != pt_stride:
        log.warning(
            f"{date_str}: mamba stride={mamba_stride} != patchtst stride={pt_stride}; "
            f"alignment between them will be APPROXIMATE."
        )

    # Use mamba's stride/window as the canonical timeline. Both prediction
    # arrays are stride-aligned from event 0 of their respective windows; we
    # truncate to common length and hope strides match (which they do for
    # bulk_oot: both 250).
    n_align = min(len(mp), len(pp))
    mp = mp[:n_align]
    ml = ml[:n_align]
    pp = pp[:n_align]
    n_preds = n_align

    # Effective stride/window we use for ts<->pred_idx mapping
    eff_window = mamba_window
    eff_stride = mamba_stride

    # Per-event arrays we need for features
    price_rel = events[:, COL_PRICE_REL].astype(np.float64)
    spread = events[:, COL_SPREAD].astype(np.float32)
    side = events[:, COL_SIDE]
    qty_log = events[:, COL_QTY_LOG]
    ts_s = timestamps_ns.astype(np.float64) / 1e9

    # Vectorized rolling volatility (window=100, same as v3)
    price_changes = np.diff(price_rel, prepend=price_rel[0])
    vol_window = 100
    cs2 = np.cumsum(price_changes ** 2)
    cs1 = np.cumsum(price_changes)
    cs2_pad = np.concatenate([[0.0], cs2])
    cs1_pad = np.concatenate([[0.0], cs1])
    idx_arr = np.arange(n_events)
    start_idx = np.maximum(idx_arr - vol_window + 1, 0)
    cnt = (idx_arr - start_idx + 1).astype(np.float64)
    sm = cs1_pad[idx_arr + 1] - cs1_pad[start_idx]
    sm2 = cs2_pad[idx_arr + 1] - cs2_pad[start_idx]
    mean_v = sm / cnt
    var_v = np.maximum(sm2 / cnt - mean_v ** 2, 0.0)
    volatility = np.sqrt(var_v).astype(np.float32)
    volatility[:vol_window] = 0.0

    # Momentum (window=50)
    mom_window = 50
    cs_p = np.cumsum(price_changes)
    cs_p_pad = np.concatenate([[0.0], cs_p])
    momentum = np.zeros(n_events, dtype=np.float32)
    valid_mom = np.arange(mom_window, n_events)
    momentum[valid_mom] = (
        (cs_p_pad[valid_mom + 1] - cs_p_pad[valid_mom - mom_window + 1]) / mom_window
    ).astype(np.float32)

    # Volume imbalance cumsums
    qty = np.exp(qty_log.astype(np.float32))
    buy_mask = (side > 0).astype(np.float32)
    sell_mask = (side < 0).astype(np.float32)
    buy_cum_pad = np.concatenate([[0.0], np.cumsum(qty * buy_mask)])
    sell_cum_pad = np.concatenate([[0.0], np.cumsum(qty * sell_mask)])
    vol_lookback = 200

    # Book / queue features (from event cols)
    bid_depth = events[:, 6] if events.shape[1] > 6 else np.ones(n_events, dtype=np.float32)
    ask_depth = events[:, 7] if events.shape[1] > 7 else np.ones(n_events, dtype=np.float32)
    book_imb = events[:, 8] if events.shape[1] > 8 else np.zeros(n_events, dtype=np.float32)
    queue_bid = events[:, 11] if events.shape[1] > 11 else np.zeros(n_events, dtype=np.float32)
    queue_ask = events[:, 12] if events.shape[1] > 12 else np.zeros(n_events, dtype=np.float32)

    # ---------------- vol_regime_bucket (within-day percentile of volatility) ----------------
    # Compute terciles of within-day rolling vol distribution; bucket each event.
    valid_vol = volatility[volatility > 0]
    if len(valid_vol) >= 100:
        vol_p33, vol_p67 = np.percentile(valid_vol, [33.3, 66.7])
    else:
        vol_p33, vol_p67 = 0.0, 0.0

    # Re-validate event count given the actual window/stride.
    if n_events < eff_window + eff_stride:
        log.warning(f"{date_str}: too few events ({n_events}) for window={eff_window} stride={eff_stride}")
        return _empty_extraction(date_str, head=head, inverse_mode=inverse_mode)

    # ---------------- Fill-sim labels ----------------
    sigma_to_pnl = load_fill_sim_labels(fillsim_path)  # signal_time_ns -> pnl_ticks
    sigma_to_pred = map_signal_ts_to_pred_idx(
        timestamps_ns, pred_stride=eff_stride, pred_window=eff_window, n_preds=n_preds
    )

    # Build labels keyed by pred_idx via NEAREST-NEIGHBOR ts match.
    # fill_sim signal_time_ns are typically quantized (100ms bars) while MBO event
    # ts are exact ns — exact equality almost never holds. Use bisect with
    # tolerance set to ~half a stride's typical wallclock duration (here 0.5s).
    TOLERANCE_NS = int(0.5 * 1e9)
    pred_event_idxs = eff_window + np.arange(n_preds) * eff_stride
    pred_event_idxs = pred_event_idxs[pred_event_idxs < n_events]
    ev_ts_at_pred = timestamps_ns[pred_event_idxs]  # sorted (events come in time order)

    pred_idx_to_label: Dict[int, float] = {}
    for ts_ns, pnl in sigma_to_pnl.items():
        j = int(np.searchsorted(ev_ts_at_pred, ts_ns))
        best = None
        best_dt = TOLERANCE_NS + 1
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(ev_ts_at_pred):
                dt = abs(int(ev_ts_at_pred[cand]) - ts_ns)
                if dt < best_dt:
                    best_dt = dt
                    best = cand
        if best is not None and best_dt <= TOLERANCE_NS:
            pred_idx_to_label[best] = pnl

    if not pred_idx_to_label:
        log.info(f"{date_str}: 0 labeled trades from {fillsim_path.name}")

    # ---------------- Iterate predictions, build feature rows for those that became trades ----------------
    feats_out = np.zeros((len(pred_idx_to_label), N_FEATURES_V4), dtype=np.float32)
    labels_out = np.zeros((len(pred_idx_to_label),), dtype=np.float32)
    meta_out = np.zeros((len(pred_idx_to_label), 3), dtype=np.float32)
    out_i = 0

    for p_idx, pnl in sorted(pred_idx_to_label.items()):
        ev_idx = eff_window + p_idx * eff_stride
        if ev_idx >= n_events:
            continue

        m1, m5, m10 = float(mp[p_idx, 0]), float(mp[p_idx, 1]), float(mp[p_idx, 2])
        if m1 == 0 and m5 == 0 and m10 == 0:
            continue
        signal_dir = 1 if m1 > 0 else -1

        abs_m1, abs_m5, abs_m10 = abs(m1), abs(m5), abs(m10)
        if abs_m1 < 0.10:
            tier = 0.0
        elif abs_m1 < 0.25:
            tier = 1.0
        elif abs_m1 < 0.50:
            tier = 2.0
        else:
            tier = 3.0

        bimb = float(book_imb[ev_idx])
        bd_log = float(np.log1p(max(float(bid_depth[ev_idx]), 0.0)))
        ad_log = float(np.log1p(max(float(ask_depth[ev_idx]), 0.0)))
        spr = float(spread[ev_idx])
        vol_v = float(volatility[ev_idx])
        ts_now = float(ts_s[ev_idx])
        tod_cont = _time_of_day_fraction(ts_now)

        sv = max(0, ev_idx - vol_lookback)
        bv = float(buy_cum_pad[ev_idx + 1] - buy_cum_pad[sv + 1])
        sv_v = float(sell_cum_pad[ev_idx + 1] - sell_cum_pad[sv + 1])
        tot = bv + sv_v
        vol_imb = (bv - sv_v) / (tot + 1e-8) if tot > 0 else 0.0

        signs_agree = (np.sign(m1) == np.sign(m5) == np.sign(m10)) and np.sign(m1) != 0
        agreement = 1.0 if signs_agree else 0.0
        strength = (abs_m1 + abs_m5 + abs_m10) / 3.0
        mom = float(momentum[ev_idx])
        preds_arr = np.array([m1, m5, m10])
        p_std = float(np.std(preds_arr))
        p_range = float(preds_arr.max() - preds_arr.min())

        # --- v4 additions ---
        p1, p5, p10 = float(pp[p_idx, 0]), float(pp[p_idx, 1]), float(pp[p_idx, 2])
        abs_p1 = abs(p1)
        sign_agree = 1.0 if (np.sign(p1) == np.sign(m1) and np.sign(p1) != 0) else 0.0
        mag_ratio = float(np.clip(abs_p1 / (abs_m1 + 1e-6), 0.0, 5.0))

        # vol_regime_bucket
        if vol_v <= vol_p33:
            vol_bucket = 0.0
        elif vol_v <= vol_p67:
            vol_bucket = 1.0
        else:
            vol_bucket = 2.0

        # tod_bucket: 0=open(<10:30 ET), 1=mid, 2=close(>15:00 ET)
        # tod_cont in [0,1] over 9:30-16:00 = 6.5h. 10:30 = (1/6.5) ≈ 0.1538; 15:00 = (5.5/6.5) ≈ 0.8462
        if tod_cont < 0.1539:
            tod_bucket = 0.0
        elif tod_cont < 0.8462:
            tod_bucket = 1.0
        else:
            tod_bucket = 2.0

        q_bid = float(queue_bid[ev_idx])
        q_ask = float(queue_ask[ev_idx])

        recent_fill_rate = 0.0  # PLACEHOLDER per HC #253(f) — TODO

        feats_out[out_i] = [
            m1, m5, m10,
            abs_m1, abs_m5, abs_m10,
            tier,
            bimb, bd_log, ad_log, spr,
            vol_v, tod_cont,
            vol_imb, agreement, strength,
            mom, p_std, p_range,
            p1, p5, p10,
            abs_p1,
            sign_agree, mag_ratio,
            vol_bucket, tod_bucket,
            q_bid, q_ask,
            recent_fill_rate,
        ]
        labels_out[out_i] = pnl
        meta_out[out_i] = [ts_now, float(p_idx), float(signal_dir)]
        out_i += 1

    feats_kept = feats_out[:out_i]
    labels_kept = labels_out[:out_i]
    meta_kept = meta_out[:out_i]

    # ----- HEAD-MODE POST-PROCESSING (HC #255) -----
    # patchtst_pred_1s lives at FEATURE_NAMES_V4.index("patchtst_pred_1s") = 19
    pt1_col = FEATURE_NAMES_V4.index("patchtst_pred_1s")
    feature_names = list(FEATURE_NAMES_V4)

    if head == HEAD_INVERSE:
        # "realized_dir" is the sign of the trade's pnl_ticks. PatchTST is "right on
        # direction" when sign(patchtst_1s) == sign(pnl_ticks). NOTE: in production
        # we'd use sign of subsequent return; we use pnl_ticks sign here because
        # that's what we have post-fill_sim and it correlates to realized direction
        # net of cost. We also separately compute it from CNN-Mamba pred for the
        # `pt_mamba_sign_agree` feature, which already exists.
        if out_i > 0:
            pt1 = feats_kept[:, pt1_col]
            pnl_sign = np.sign(labels_kept).astype(np.float32)
            pt_sign = np.sign(pt1).astype(np.float32)
            sign_correct = (pt_sign == pnl_sign) & (pt_sign != 0)

            if inverse_mode == INVERSE_MODE_PREGATE:
                # Drop rows where PatchTST got direction wrong (or was zero).
                keep = sign_correct
                feats_kept = feats_kept[keep]
                labels_kept = labels_kept[keep]
                meta_kept = meta_kept[keep]
                # feature_names unchanged
            else:  # INVERSE_MODE_FEATURE
                extra = sign_correct.astype(np.float32).reshape(-1, 1)
                feats_kept = np.concatenate([feats_kept, extra], axis=1)
                feature_names = feature_names + ["patchtst_sign_correct"]
        else:
            # No labeled rows; in feature-mode we still want the column to exist
            if inverse_mode == INVERSE_MODE_FEATURE:
                feats_kept = np.zeros((0, N_FEATURES_V4 + 1), dtype=np.float32)
                feature_names = feature_names + ["patchtst_sign_correct"]

    return {
        "features": feats_kept.copy(),
        "labels": labels_kept.copy(),
        "meta": meta_kept.copy(),
        "feature_names": feature_names,
        "n_predictions_total": int(n_preds),
        "n_labeled": int(len(labels_kept)),
        "date_str": date_str,
        "head": head,
        "inverse_mode": inverse_mode,
    }


def _empty_extraction(
    date_str: str,
    head: str = HEAD_CONFLUENCE,
    inverse_mode: str = INVERSE_MODE_FEATURE,
) -> Dict[str, np.ndarray]:
    n_feats = N_FEATURES_V4 + (1 if (head == HEAD_INVERSE and inverse_mode == INVERSE_MODE_FEATURE) else 0)
    feature_names = list(FEATURE_NAMES_V4)
    if head == HEAD_INVERSE and inverse_mode == INVERSE_MODE_FEATURE:
        feature_names = feature_names + ["patchtst_sign_correct"]
    return {
        "features": np.zeros((0, n_feats), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.float32),
        "meta": np.zeros((0, 3), dtype=np.float32),
        "feature_names": feature_names,
        "n_predictions_total": 0,
        "n_labeled": 0,
        "date_str": date_str,
        "head": head,
        "inverse_mode": inverse_mode,
    }


# ===========================================================================
# Smoke test — load 1 OOT date, build features+labels, print sanity stats
# ===========================================================================

def smoke_test(
    date_str: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    mamba_dir: Path = DEFAULT_MAMBA_DIR,
    patchtst_dir: Path = DEFAULT_PATCHTST_DIR,
    fillsim_dir: Path = DEFAULT_FILLSIM_DIR,
    fillsim_config: str = DEFAULT_FILLSIM_CONFIG,
    head: str = HEAD_CONFLUENCE,
    inverse_mode: str = INVERSE_MODE_FEATURE,
) -> Dict:
    """End-to-end on one date. Print shapes + sanity stats. Do NOT train."""
    log.info("=" * 70)
    log.info(
        f"SMOKE TEST — date {date_str} | fill_sim cfg {fillsim_config} | "
        f"head={head} inverse_mode={inverse_mode}"
    )
    log.info("=" * 70)
    log.info(f"FEATURE_NAMES_V4 ({N_FEATURES_V4} features):")
    for i, n in enumerate(FEATURE_NAMES_V4):
        log.info(f"  [{i:2d}] {n}")

    mbo_path = data_dir / f"{date_str}_mbo_events.npz"
    mamba_path = mamba_dir / f"{date_str}_predictions.npz"
    pt_path = patchtst_dir / f"{date_str}_predictions.npz"
    fs_path = fillsim_dir / f"{fillsim_config}_{date_str}.json"

    for p, name in [(mbo_path, "MBO"), (mamba_path, "CNN-Mamba"),
                    (pt_path, "PatchTST"), (fs_path, "fill_sim")]:
        ok = "OK" if p.exists() else "MISSING"
        log.info(f"  {name}: {p}  [{ok}]")

    if not (mbo_path.exists() and mamba_path.exists() and pt_path.exists()):
        log.error("Missing critical inputs. Aborting smoke test.")
        return {}

    t0 = time.time()
    res = extract_v4_for_date(
        mbo_path, mamba_path, pt_path, fs_path,
        head=head, inverse_mode=inverse_mode,
    )
    elapsed = time.time() - t0

    feature_names = res.get("feature_names", FEATURE_NAMES_V4)
    n_feats = res["features"].shape[1] if res["features"].size else len(feature_names)

    log.info(
        f"\nResults [head={res.get('head')} inverse_mode={res.get('inverse_mode')}]:\n"
        f"  shape(features) = {res['features'].shape}  (n_feats={n_feats})\n"
        f"  shape(labels)   = {res['labels'].shape}\n"
        f"  shape(meta)     = {res['meta'].shape}\n"
        f"  n_predictions   = {res['n_predictions_total']}\n"
        f"  n_labeled       = {res['n_labeled']}\n"
        f"  elapsed         = {elapsed:.2f}s"
    )

    if res["n_labeled"] > 0:
        feats = res["features"]
        labels = res["labels"]
        log.info("\nFeature sanity (mean / std / min / max per column):")
        for i, name in enumerate(feature_names):
            col = feats[:, i]
            log.info(
                f"  [{i:2d}] {name:>24s}  "
                f"mean={col.mean():+.4f}  std={col.std():.4f}  "
                f"min={col.min():+.4f}  max={col.max():+.4f}"
            )
        log.info(
            f"\nLabel (pnl_ticks) sanity:  "
            f"mean={labels.mean():+.4f}  std={labels.std():.4f}  "
            f"min={labels.min():+.4f}  max={labels.max():+.4f}  "
            f"WR={float((labels > 0).mean()):.3f}"
        )
        log.info(
            f"  net mean (commission already in label): {labels.mean():+.4f} ticks  "
            f"(commission constant: {COMMISSION_TICKS:.3f}t)"
        )

        # Top-3 |corr(feature, target)|
        if labels.std() > 1e-9:
            corrs = []
            for i, name in enumerate(feature_names):
                col = feats[:, i]
                if col.std() > 1e-9:
                    c = float(np.corrcoef(col, labels)[0, 1])
                else:
                    c = 0.0
                corrs.append((name, c))
            corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
            log.info("\nTop-3 |corr(feature, pnl_ticks)|:")
            for name, c in corrs_sorted[:3]:
                log.info(f"  {name:>24s}  corr={c:+.4f}")
        else:
            log.info("\nTop-3 corr: SKIPPED (label std ~ 0)")

        # Single-line summary (easy to grep)
        log.info(
            f"\nSMOKE_SUMMARY  head={res.get('head')}  inverse_mode={res.get('inverse_mode')}  "
            f"n_trades={res['n_labeled']}  n_features={n_feats}  "
            f"label_mean={labels.mean():+.4f}  label_std={labels.std():.4f}  "
            f"WR={float((labels > 0).mean()):.3f}"
        )
    else:
        log.info(
            f"\nSMOKE_SUMMARY  head={res.get('head')}  inverse_mode={res.get('inverse_mode')}  "
            f"n_trades=0  n_features={n_feats}"
        )
    return res


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Build supervised exec v4 features + labels (smoke test)")
    ap.add_argument("--smoke-date", type=str, default="20260415",
                    help="OOT date YYYYMMDD for smoke test")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--mamba-dir", type=str, default=str(DEFAULT_MAMBA_DIR))
    ap.add_argument("--patchtst-dir", type=str, default=str(DEFAULT_PATCHTST_DIR))
    ap.add_argument("--fillsim-dir", type=str, default=str(DEFAULT_FILLSIM_DIR))
    ap.add_argument("--fillsim-config", type=str, default=DEFAULT_FILLSIM_CONFIG)
    ap.add_argument(
        "--head", choices=[HEAD_CONFLUENCE, HEAD_INVERSE], default=HEAD_CONFLUENCE,
        help="confluence: CNN-Mamba sign decides direction (default v4). "
             "inverse: PatchTST sign decides direction; CNN-Mamba is magnitude/edge (HC #255).",
    )
    ap.add_argument(
        "--inverse-mode",
        choices=[INVERSE_MODE_PREGATE, INVERSE_MODE_FEATURE],
        default=INVERSE_MODE_FEATURE,
        help="Only used when --head=inverse. "
             "pregate: drop trades where PatchTST got direction wrong. "
             "feature: keep all trades, append patchtst_sign_correct binary feature.",
    )
    ap.add_argument(
        "--smoke-both", action="store_true",
        help="Run smoke test on confluence head AND both inverse-mode variants on the "
             "same date. Overrides --head / --inverse-mode for the smoke run.",
    )
    args = ap.parse_args()

    common = dict(
        date_str=args.smoke_date,
        data_dir=Path(args.data_dir),
        mamba_dir=Path(args.mamba_dir),
        patchtst_dir=Path(args.patchtst_dir),
        fillsim_dir=Path(args.fillsim_dir),
        fillsim_config=args.fillsim_config,
    )

    if args.smoke_both:
        log.info("\n### SMOKE-BOTH: testing all heads on same date ###")
        smoke_test(head=HEAD_CONFLUENCE, inverse_mode=INVERSE_MODE_FEATURE, **common)
        smoke_test(head=HEAD_INVERSE, inverse_mode=INVERSE_MODE_PREGATE, **common)
        smoke_test(head=HEAD_INVERSE, inverse_mode=INVERSE_MODE_FEATURE, **common)
    else:
        smoke_test(head=args.head, inverse_mode=args.inverse_mode, **common)


if __name__ == "__main__":
    main()
