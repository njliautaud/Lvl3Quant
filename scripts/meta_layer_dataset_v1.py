#!/usr/bin/env python3
"""
meta_layer_dataset_v1.py — Pre-compute the training dataset for the HC #486 R4
meta-layer classifier (stream + raw-data window -> entry/exit decision).

This is INFRASTRUCTURE, not a hypothesis test. Success = clean dataset produced.

INPUTS (per OOT date present in cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate):
  - v3.4.2 OOT predictions: per-event pred_log_ret_{1s,5s,10s,30s}, target_log_ret_*
      shape ~ (49530,) per day, stride=250 events, window=1500.
      event_idx = pred_idx * 250 + 1499
  - Raw MBO events: data/processed/mbo_events/{YYYYMMDD}_mbo_events.npz
      events[:,0]=time_delta_log, [:,1]=event_type, [:,2]=side, [:,3]=price_rel_ticks,
      [:,4]=qty_log, [:,5]=spread_ticks ; shape ~ (12.4M, 6)
  - OFI features: data/processed/mbo_events_smart_v3_ofi_features/{YYYYMMDD}_ofi.npz
      Per-event ofi_aggressive_{1s,5s,10s,30s}, ofi_book_{1s,5s,10s,30s},
      trade_signed_flow_{1s,5s,10s,30s}, spread_ticks_now ; shape == raw mbo.

OUTPUTS (per date):
  /home/jupiter/Lvl3Quant/data/processed/meta_layer_v1/{YYYYMMDD}_meta.npz
  Contains:
    X (N, n_feat) float32 — stream-window + raw-data-window features
    y (N, 16) float32 — 8 net-tick + 8 binary labels (4 horizons * 2 sides)
    feat_names (n_feat,) <U64
    label_names (16,) <U32
    pred_idx (N,) int64 (which prediction-stream index)
    event_idx (N,) int64 (mapped raw mbo event index)
    date (1,) <U8

Plus:
  /home/jupiter/Lvl3Quant/data/processed/meta_layer_v1/meta_layer_v1_manifest.json
  Manifest per HC #485 R5.

CONSTRAINTS:
  - NO LOOK-AHEAD in X. Stream-window features use t..t (current event only,
    NOT t+1..t+K — see HC #486 R3). Raw-data window features use t-W..t-1.
  - Future-only data (target_log_ret_*) used ONLY in y.
  - Pure CPU numpy/pandas. Target <30 min total.
  - Per HC #485 R1: per-date NaN fraction logged per output column.
  - Per HC #485 R3: post-regen summary; fail if avg NaN >0.10 on any column.
  - Per HC #420: this is the user's own quant research codebase.

ENTRY-COST: per CLAUDE.md cost constants, passive-limit entry = 0.376 ticks.
  Convert log_ret -> ticks for ES futures (1 tick = 0.25 pts, mid ~ 5000):
  ticks ~ (exp(log_ret) - 1) * 5000 / 0.25 ~= log_ret * 5000 / 0.25 = log_ret * 20000
  for small log_ret. We'll use this linearization since |log_ret| << 1.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
RAW_MBO_DIR = ROOT / "data/processed/mbo_events"
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
OUT_DIR = ROOT / "data/processed/meta_layer_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v3.4.2 stride / window (verified from live_trading_linux/v3_4_2_inference.py)
WINDOW = 1500
STRIDE = 250

# Stream-window K values (events in prediction stride space; 1 K = 250ms)
STREAM_K_VALUES = [4, 20, 40]            # sign_consistency
STREAM_DRIFT_K = 20
STREAM_FLIP_K = 20
STREAM_VAR_K = 20
STREAM_MEANABS_K = 20

# Raw-data window (events in MBO stride space — STRICTLY PAST W events)
RAW_W_EVENTS = 20

# ES tick / cost
# IMPORTANT: despite the column name target_log_ret_*, the v3.4.2 OOT targets
# are stored in TICKS (integer-valued; std ~1.6 t at 1s, ~8 t at 30s).
# Verified by inspection 2026-05-22. No conversion needed.
TARGET_IS_TICKS = True
PASSIVE_LIMIT_COST_TICKS = 0.376  # AMP commission only (HC: CLAUDE.md COST CONSTANTS)

HORIZONS = ["1s", "5s", "10s", "30s"]

# Event-type / side codes (per ofi_features_v1.py)
ET_ADD, ET_CXL, ET_MOD, ET_TRADE, ET_FILL = 0, 1, 2, 3, 4
SIDE_BID, SIDE_ASK, SIDE_NONE = 0, 1, 2

NAN_FAIL_THRESHOLD = 0.10


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ----------------------------------------------------------------------------
# Stream-window features (causal — current + past predictions only)
# ----------------------------------------------------------------------------
def _trailing_window_op(x: np.ndarray, K: int, op):
    """For each i, op(x[max(0,i-K+1):i+1]). Returns array shape (N,)."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        win = x[lo:i + 1]
        if len(win) < 2:
            continue
        out[i] = op(win)
    return out


def _trailing_sign_consistency_fast(x: np.ndarray, K: int) -> np.ndarray:
    """Fraction of prior K (inclusive of current) preds with same sign as current.
    Causal: uses x[i-K+1..i] for output at i."""
    n = len(x)
    sign_x = np.sign(x).astype(np.float32)
    # rolling sum of sign via cumsum
    cs = np.concatenate(([0.0], np.cumsum(sign_x, dtype=np.float64)))
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        hi = i + 1
        wlen = hi - lo
        if wlen < 2:
            continue
        # mean of sign in window
        mean_sign = (cs[hi] - cs[lo]) / wlen
        cur_sign = sign_x[i]
        if cur_sign == 0:
            out[i] = 0.0
            continue
        # fraction agreeing with current sign
        # mean_sign = (n_pos - n_neg) / wlen; n_agree = wlen * (1 + cur_sign*mean_sign) / 2
        out[i] = float(0.5 * (1.0 + cur_sign * mean_sign))
    return out


def _trailing_cumdrift_fast(x: np.ndarray, K: int) -> np.ndarray:
    """Sum of x[i-K+1..i] (cumulative drift over trailing K, including current)."""
    n = len(x)
    cs = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        hi = i + 1
        if hi - lo < 2:
            continue
        out[i] = float(cs[hi] - cs[lo])
    return out


def _trailing_flip_rate_fast(x: np.ndarray, K: int) -> np.ndarray:
    """Fraction of adjacent-pair sign flips in x[i-K+1..i]."""
    n = len(x)
    sign_x = np.sign(x).astype(np.int8)
    flips = np.zeros(n, dtype=np.int32)
    flips[1:] = (sign_x[1:] != sign_x[:-1]).astype(np.int32)
    cs = np.concatenate(([0], np.cumsum(flips)))
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        hi = i + 1
        # number of adjacencies inside window = (hi - lo - 1)
        # number of flips inside window = flips[lo+1..hi-1].sum() = cs[hi] - cs[lo+1]
        wlen = hi - lo
        if wlen < 2:
            continue
        n_adj = wlen - 1
        n_flips = cs[hi] - cs[lo + 1]
        out[i] = float(n_flips) / float(n_adj)
    return out


def _trailing_var_fast(x: np.ndarray, K: int) -> np.ndarray:
    """Rolling variance over trailing K events including current."""
    n = len(x)
    xf = x.astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(xf)))
    cs2 = np.concatenate(([0.0], np.cumsum(xf * xf)))
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        hi = i + 1
        wlen = hi - lo
        if wlen < 2:
            continue
        m1 = (cs[hi] - cs[lo]) / wlen
        m2 = (cs2[hi] - cs2[lo]) / wlen
        v = max(0.0, m2 - m1 * m1)
        out[i] = float(v)
    return out


def _trailing_mean_abs_fast(x: np.ndarray, K: int) -> np.ndarray:
    n = len(x)
    ax = np.abs(x).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(ax)))
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - K + 1)
        hi = i + 1
        wlen = hi - lo
        if wlen < 2:
            continue
        out[i] = float((cs[hi] - cs[lo]) / wlen)
    return out


# ----------------------------------------------------------------------------
# Raw-data window features (strictly past W events at MBO resolution)
# ----------------------------------------------------------------------------
def _past_window_mean_at(values: np.ndarray, event_idx: np.ndarray, W: int) -> np.ndarray:
    """For each (sampled) event index i in event_idx, return mean of values[i-W..i-1].
    Strictly PAST (excludes current event). NaN if i < W."""
    cs = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
    n_sampled = len(event_idx)
    out = np.full(n_sampled, np.nan, dtype=np.float32)
    for k in range(n_sampled):
        i = int(event_idx[k])
        lo = i - W
        hi = i
        if lo < 0:
            continue
        out[k] = float((cs[hi] - cs[lo]) / W)
    return out


def _past_window_sum_at(values: np.ndarray, event_idx: np.ndarray, W: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
    n_sampled = len(event_idx)
    out = np.full(n_sampled, np.nan, dtype=np.float32)
    for k in range(n_sampled):
        i = int(event_idx[k])
        lo = i - W
        hi = i
        if lo < 0:
            continue
        out[k] = float(cs[hi] - cs[lo])
    return out


def _past_window_drift_ticks_at(price_rel_ticks: np.ndarray, event_idx: np.ndarray, W: int) -> np.ndarray:
    """Mid-price drift over past W events in ticks. Approximated as
    price_rel_ticks[i-1] - price_rel_ticks[i-W]. NaN if i < W."""
    n_sampled = len(event_idx)
    out = np.full(n_sampled, np.nan, dtype=np.float32)
    pr = price_rel_ticks
    for k in range(n_sampled):
        i = int(event_idx[k])
        if i - W < 0 or i - 1 < 0:
            continue
        out[k] = float(pr[i - 1] - pr[i - W])
    return out


# ----------------------------------------------------------------------------
# Per-date builder
# ----------------------------------------------------------------------------
def build_for_date(date_str: str) -> dict | None:
    pred_path = PRED_DIR / f"oot_{date_str}.npz"
    raw_path = RAW_MBO_DIR / f"{date_str}_mbo_events.npz"
    ofi_path = OFI_DIR / f"{date_str}_ofi.npz"
    if not (pred_path.exists() and raw_path.exists() and ofi_path.exists()):
        log(f"  SKIP {date_str}: missing inputs (pred={pred_path.exists()} raw={raw_path.exists()} ofi={ofi_path.exists()})")
        return None

    # ----- load -----
    p = np.load(pred_path, allow_pickle=True)
    r = np.load(raw_path, allow_pickle=True)
    o = np.load(ofi_path, allow_pickle=True)

    pred_1s = p["pred_log_ret_1s"].astype(np.float32)
    pred_5s = p["pred_log_ret_5s"].astype(np.float32)
    pred_10s = p["pred_log_ret_10s"].astype(np.float32)
    pred_30s = p["pred_log_ret_30s"].astype(np.float32)
    tgt_1s = p["target_log_ret_1s"].astype(np.float32)
    tgt_5s = p["target_log_ret_5s"].astype(np.float32)
    tgt_10s = p["target_log_ret_10s"].astype(np.float32)
    tgt_30s = p["target_log_ret_30s"].astype(np.float32)
    N = len(pred_1s)

    events = r["events"]
    n_events = events.shape[0]
    price_rel_ticks = events[:, 3].astype(np.float32)
    spread_ticks = events[:, 5].astype(np.float32)
    et = events[:, 1].astype(np.int8)
    side = events[:, 2].astype(np.int8)
    qty_log = events[:, 4].astype(np.float32)
    size = np.exp(qty_log.astype(np.float64))

    # signed trade flow per event (matches ofi_features_v1.py convention)
    trade_signed = np.zeros(n_events, dtype=np.float64)
    is_trade = (et == ET_TRADE)
    trade_signed[is_trade & (side == SIDE_ASK)] = size[is_trade & (side == SIDE_ASK)]
    trade_signed[is_trade & (side == SIDE_BID)] = -size[is_trade & (side == SIDE_BID)]

    # queue-imbalance proxy: bid-book minus ask-book ADDs/CXLs per event
    # (true BB/BA-size queue imbalance unavailable per ofi_features_v1.py NOTE.)
    qi_event = np.zeros(n_events, dtype=np.float64)
    is_add = (et == ET_ADD)
    is_cxl = (et == ET_CXL)
    qi_event[is_add & (side == SIDE_BID)] = size[is_add & (side == SIDE_BID)]
    qi_event[is_cxl & (side == SIDE_BID)] = -size[is_cxl & (side == SIDE_BID)]
    qi_event[is_add & (side == SIDE_ASK)] = -size[is_add & (side == SIDE_ASK)]
    qi_event[is_cxl & (side == SIDE_ASK)] = size[is_cxl & (side == SIDE_ASK)]

    ofi_5s = o["ofi_aggressive_5s"].astype(np.float32)
    ofi_10s = o["ofi_aggressive_10s"].astype(np.float32)

    # ----- pred-idx -> event-idx map -----
    pred_idx = np.arange(N, dtype=np.int64)
    event_idx = pred_idx * STRIDE + (WINDOW - 1)
    if event_idx.max() >= n_events:
        # clip safely (should not happen given stride/window math)
        valid = event_idx < n_events
        log(f"  WARN {date_str}: {(~valid).sum()} predictions exceed n_events; clipping")
        event_idx = np.minimum(event_idx, n_events - 1)

    # ----- (1) Prediction-stream window features (CAUSAL — past + current) -----
    feats: dict[str, np.ndarray] = {}

    for K in STREAM_K_VALUES:
        feats[f"sign_consistency_K{K}"] = _trailing_sign_consistency_fast(pred_1s, K)
    feats[f"cum_drift_K{STREAM_DRIFT_K}"] = _trailing_cumdrift_fast(pred_1s, STREAM_DRIFT_K)
    feats[f"flip_rate_K{STREAM_FLIP_K}"] = _trailing_flip_rate_fast(pred_1s, STREAM_FLIP_K)
    feats[f"rolling_var_K{STREAM_VAR_K}"] = _trailing_var_fast(pred_1s, STREAM_VAR_K)
    feats[f"mean_abs_pred_K{STREAM_MEANABS_K}"] = _trailing_mean_abs_fast(pred_1s, STREAM_MEANABS_K)

    # current pred at each horizon
    feats["pred_log_ret_1s"] = pred_1s
    feats["pred_log_ret_5s"] = pred_5s
    feats["pred_log_ret_10s"] = pred_10s
    feats["pred_log_ret_30s"] = pred_30s

    # multi-h sign agreement (1 if same nonzero sign, 0 otherwise; sign(0)=0 treated as no-agreement)
    def _sign_agree(a, b):
        sa = np.sign(a)
        sb = np.sign(b)
        return ((sa == sb) & (sa != 0)).astype(np.float32)
    feats["sign_agree_1s_5s"] = _sign_agree(pred_1s, pred_5s)
    feats["sign_agree_5s_10s"] = _sign_agree(pred_5s, pred_10s)
    feats["sign_agree_10s_30s"] = _sign_agree(pred_10s, pred_30s)

    # ----- (2) Raw-data window features (STRICTLY PAST W events at MBO resolution) -----
    feats["mean_spread_W20"] = _past_window_mean_at(spread_ticks, event_idx, RAW_W_EVENTS)
    feats["mean_queue_imb_W20"] = _past_window_sum_at(qi_event, event_idx, RAW_W_EVENTS) / float(RAW_W_EVENTS)
    feats["signed_trade_flow_W20"] = _past_window_sum_at(trade_signed, event_idx, RAW_W_EVENTS)
    feats["mid_price_drift_ticks_W20"] = _past_window_drift_ticks_at(price_rel_ticks, event_idx, RAW_W_EVENTS)

    # current OFI 5s / 10s (causal-by-construction in ofi_features_v1.py — strictly prior W seconds)
    feats["ofi_5s_now"] = ofi_5s[event_idx].astype(np.float32)
    feats["ofi_10s_now"] = ofi_10s[event_idx].astype(np.float32)

    # ----- order features in canonical order -----
    feat_order = (
        [f"sign_consistency_K{K}" for K in STREAM_K_VALUES]
        + [f"cum_drift_K{STREAM_DRIFT_K}",
           f"flip_rate_K{STREAM_FLIP_K}",
           f"rolling_var_K{STREAM_VAR_K}",
           f"mean_abs_pred_K{STREAM_MEANABS_K}",
           "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s",
           "sign_agree_1s_5s", "sign_agree_5s_10s", "sign_agree_10s_30s",
           "mean_spread_W20", "mean_queue_imb_W20", "signed_trade_flow_W20",
           "mid_price_drift_ticks_W20", "ofi_5s_now", "ofi_10s_now"]
    )
    X = np.stack([feats[c].astype(np.float32) for c in feat_order], axis=1)

    # ----- (3) Labels — net ticks at PASSIVE-LIMIT entry per horizon, both sides -----
    horizon_tgts = {"1s": tgt_1s, "5s": tgt_5s, "10s": tgt_10s, "30s": tgt_30s}
    label_cols = {}
    label_order = []
    for h in HORIZONS:
        tgt = horizon_tgts[h]
        ticks_realized = tgt  # already in ticks
        y_long_net = ticks_realized - PASSIVE_LIMIT_COST_TICKS
        y_short_net = -ticks_realized - PASSIVE_LIMIT_COST_TICKS
        label_cols[f"y_long_{h}_net"] = y_long_net.astype(np.float32)
        label_cols[f"y_short_{h}_net"] = y_short_net.astype(np.float32)
        label_order += [f"y_long_{h}_net", f"y_short_{h}_net"]
    for h in HORIZONS:
        long_net = label_cols[f"y_long_{h}_net"]
        short_net = label_cols[f"y_short_{h}_net"]
        # propagate NaN: winner = NaN if net is NaN, else 1/0
        lw = np.where(np.isfinite(long_net), (long_net > 0).astype(np.float32), np.nan).astype(np.float32)
        sw = np.where(np.isfinite(short_net), (short_net > 0).astype(np.float32), np.nan).astype(np.float32)
        label_cols[f"y_long_{h}_winner"] = lw
        label_cols[f"y_short_{h}_winner"] = sw
        label_order += [f"y_long_{h}_winner", f"y_short_{h}_winner"]

    y = np.stack([label_cols[c] for c in label_order], axis=1)

    # ----- NaN audit per output column (HC #485 R1) -----
    col_nan_frac = {}
    for j, c in enumerate(feat_order):
        col_nan_frac[c] = float(np.mean(~np.isfinite(X[:, j])))
    for j, c in enumerate(label_order):
        col_nan_frac[c] = float(np.mean(~np.isfinite(y[:, j])))

    # ----- save -----
    out_path = OUT_DIR / f"{date_str}_meta.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        feat_names=np.array(feat_order, dtype="<U64"),
        label_names=np.array(label_order, dtype="<U32"),
        pred_idx=pred_idx.astype(np.int64),
        event_idx=event_idx.astype(np.int64),
        date=np.array([date_str], dtype="<U8"),
    )
    return {
        "date": date_str,
        "n_rows": int(N),
        "n_feat": int(X.shape[1]),
        "n_label": int(y.shape[1]),
        "out": str(out_path),
        "col_nan_frac": col_nan_frac,
        "feat_order": feat_order,
        "label_order": label_order,
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit dates for testing")
    ap.add_argument("--dates", type=str, default="", help="comma-sep YYYYMMDD list")
    args = ap.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    log(f"[start] meta_layer_dataset_v1  started_at={started_at}")
    log(f"[paths] PRED={PRED_DIR}  RAW={RAW_MBO_DIR}  OFI={OFI_DIR}  OUT={OUT_DIR}")

    if args.dates:
        dates = sorted(args.dates.split(","))
    else:
        pred_files = sorted(PRED_DIR.glob("oot_*.npz"))
        dates = [p.stem.replace("oot_", "") for p in pred_files]
        # only keep dates that have raw + ofi
        keep = []
        for d in dates:
            if (RAW_MBO_DIR / f"{d}_mbo_events.npz").exists() and (OFI_DIR / f"{d}_ofi.npz").exists():
                keep.append(d)
        dates = keep
    if args.limit > 0:
        dates = dates[: args.limit]

    log(f"[plan] {len(dates)} dates  first={dates[:2]}  last={dates[-2:] if len(dates) >= 2 else dates}")

    per_date_results = []
    n_corrupt = 0
    for i, d in enumerate(dates, start=1):
        try:
            t_d = time.time()
            res = build_for_date(d)
            if res is None:
                n_corrupt += 1
                continue
            elapsed_d = time.time() - t_d
            # log per-date NaN summary (HC #485 R1)
            worst = sorted(res["col_nan_frac"].items(), key=lambda kv: -kv[1])[:3]
            worst_str = ", ".join([f"{k}={v:.3f}" for k, v in worst])
            log(f"  [{i}/{len(dates)}] {d}: N={res['n_rows']:,} feat={res['n_feat']} label={res['n_label']} "
                f"worst_nan=[{worst_str}] dt={elapsed_d:.1f}s")
            per_date_results.append(res)
        except Exception as e:
            import traceback
            log(f"  FAILED {d}: {e}\n{traceback.format_exc()}")
            n_corrupt += 1

    # ----- aggregate manifest (HC #485 R3 + R5) -----
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_events = int(sum(r["n_rows"] for r in per_date_results))

    # worst NaN frac per column (max across dates) AND avg NaN frac per column
    all_cols = set()
    for r in per_date_results:
        all_cols.update(r["col_nan_frac"].keys())
    worst_nan_frac = {}
    avg_nan_frac = {}
    for c in all_cols:
        vals = [r["col_nan_frac"].get(c, np.nan) for r in per_date_results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            worst_nan_frac[c] = None
            avg_nan_frac[c] = None
        else:
            worst_nan_frac[c] = float(max(vals))
            avg_nan_frac[c] = float(np.mean(vals))

    # FAIL gate (HC #485 R3): avg NaN > 0.10 on any column
    bad_cols = {c: v for c, v in avg_nan_frac.items() if v is not None and v > NAN_FAIL_THRESHOLD}
    pass_nan_gate = len(bad_cols) == 0

    manifest = {
        "task": "meta_layer_dataset_v1",
        "hc_refs": ["HC#486R4", "HC#486R3", "HC#485R1", "HC#485R3", "HC#485R5", "HC#420"],
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(time.time() - t0, 1),
        "n_files_in": len(dates),
        "n_files_out": len(per_date_results),
        "n_corrupt": n_corrupt,
        "total_events": total_events,
        "feature_count": per_date_results[0]["n_feat"] if per_date_results else 0,
        "label_count": per_date_results[0]["n_label"] if per_date_results else 0,
        "feat_names": per_date_results[0]["feat_order"] if per_date_results else [],
        "label_names": per_date_results[0]["label_order"] if per_date_results else [],
        "worst_nan_frac": worst_nan_frac,
        "avg_nan_frac": avg_nan_frac,
        "nan_fail_threshold": NAN_FAIL_THRESHOLD,
        "pass_nan_gate": pass_nan_gate,
        "failing_columns_avg_gt_threshold": bad_cols,
        "constants": {
            "WINDOW": WINDOW,
            "STRIDE": STRIDE,
            "STREAM_K_VALUES": STREAM_K_VALUES,
            "RAW_W_EVENTS": RAW_W_EVENTS,
            "TARGET_IS_TICKS": TARGET_IS_TICKS,
            "PASSIVE_LIMIT_COST_TICKS": PASSIVE_LIMIT_COST_TICKS,
        },
        "fix_commit_sha": "meta_layer_v1",
        "out_dir": str(OUT_DIR),
    }
    manifest_path = OUT_DIR / "meta_layer_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    log(f"[manifest] wrote {manifest_path}")
    log(f"[summary] dates_out={len(per_date_results)}/{len(dates)} corrupt={n_corrupt} "
        f"total_events={total_events:,} elapsed={time.time()-t0:.1f}s")
    log(f"[nan_gate] pass={pass_nan_gate}  bad_cols={list(bad_cols.keys())}")
    if not pass_nan_gate:
        log(f"[nan_gate] FAILING (avg NaN > {NAN_FAIL_THRESHOLD}):")
        for c, v in bad_cols.items():
            log(f"           {c}: avg_nan={v:.4f}")
        sys.exit(2)
    log(f"[done]")


if __name__ == "__main__":
    main()
