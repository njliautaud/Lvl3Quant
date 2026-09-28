#!/usr/bin/env python3
"""es_spy_lead_lag_ic.py — ES CNN-Mamba v3.4.2 -> SPY lead-lag IC study.

REFACTORED 2026-06-04 per HC #529 to use the PRODUCTION CHAMPION v3.4.2
weights (was v2 which is now deprecated).

Model: CNN-Mamba v3.4.2 (fold_03_intra_ckpt.pt from cnn_mamba_v3_4_2_hc477fix_v2)
  - Architecture: CNNMambaV32 trunk (the v3.4.2 ckpt has no book_cnn keys --
    book pathway was gated to 0 at training, so trunk alone reproduces it).
  - Primary directional head: log_ret_5s (concat IC ~0.108-0.141 in OOT folds).
    log_ret_1s and log_ret_10s also emitted (for cross-horizon sanity).

Per SPY_EXECUTION_ONLY_RESEARCH.md §5:
  For each ES prediction at ts_pred, for each (lag, horizon) pair,
  compute realized SPY mid forward-return over [ts_pred+lag, ts_pred+lag+h].
  Then rank IC of (ES_pred, SPY_fwd_ret) per (lag, h), per day, and concat.

Outputs (under OUTPUT_DIR):
  lag_ic_matrix.json -- full IC matrix incl. per-day breakdown + n samples
  lag_ic_heatmap.png -- heatmap visual
  REPORT.md          -- plain-English summary + decision
  preds_<date>.npz   -- cached ES predictions per date (v3.4.2 schema)
  spy_mid_<date>.npz -- cached SPY mid time-series per date (KEPT from v2 run)

Designed to run on Neptune (RTX 3090) inside ~/training-env:
  source ~/training-env/bin/activate
  cd /home/nick/Lvl3Quant
  PYTHONPATH=/home/nick/Lvl3Quant OUTPUT_DIR=/home/nick/Lvl3Quant/output/spy_exec_only \\
    python3 scripts/spy_trial/es_spy_lead_lag_ic.py
"""
from __future__ import annotations

import os
# Set BEFORE importing v3_2 trainer (which reads these via os.environ)
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
os.environ.setdefault("EVENT_STRIDE", "250")

import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Path bootstrap so we can import alpha_discovery on Neptune
_REPO_ROOT = Path(os.environ.get("LVL3_REPO", "/home/nick/Lvl3Quant"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATES = [
    "20260302", "20260303", "20260304", "20260305", "20260306",
    "20260309", "20260310", "20260311", "20260312",
]

ES_EVENTS_DIR = Path(os.environ.get(
    "ES_EVENTS_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3",
))
SPY_RAW_DIR = Path(os.environ.get(
    "SPY_RAW_DIR",
    "/home/nick/Lvl3Quant/data/raw/spy_mbo",
))
SPY_EVENTS_DIR = Path(os.environ.get(
    "SPY_EVENTS_DIR",
    "/home/nick/Lvl3Quant/data/processed/spy_mbo_events",
))

# v3.4.2 production champion (HC #529)
WEIGHTS_PATH = Path(os.environ.get(
    "WEIGHTS_PATH",
    "/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2/fold_03_intra_ckpt.pt",
))
STATS_PATH = Path(os.environ.get(
    "STATS_PATH",
    "/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2/fold_03_feature_stats.npz",
))

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "/home/nick/Lvl3Quant/output/spy_exec_only",
))
LOCAL_CACHE_DIR = Path(os.environ.get(
    "LOCAL_CACHE_DIR",
    "/home/nick/Lvl3Quant/output/spy_exec_only",
))

# Aux dirs needed by SmartV32Dataset (we don't use labels but it scans for them)
PT_PRED_DIR = Path(os.environ.get(
    "PT_PRED_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_pt_pred",
))
FIFO_LABEL_DIR = Path(os.environ.get(
    "FIFO_LABEL_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels",
))
ALPHA_LABEL_DIR = Path(os.environ.get(
    "ALPHA_LABEL_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v2",
))
TIER2_PARQUET = Path(os.environ.get(
    "TIER2_PARQUET",
    "/home/nick/Lvl3Quant/data/derived/tier2_orderflow_features_v1.parquet",
))
TIER3_PARQUET = Path(os.environ.get(
    "TIER3_PARQUET",
    "/home/nick/Lvl3Quant/data/derived/tier3_session_features_v1.parquet",
))

# Inference batch size (v3.4.2 model is ~1.5M params, can go bigger)
BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 128))

# Lag in ms, horizon in seconds
LAGS_MS = [0, 50, 100, 200, 500, 1000]
HORIZONS_S = [1, 5, 10]

# RTH (UTC) per date -- pre vs post-DST. DST started Sun Mar 8 2026.
RTH_BY_DATE = {
    "20260302": (14 * 3600 + 30 * 60, 21 * 3600),
    "20260303": (14 * 3600 + 30 * 60, 21 * 3600),
    "20260304": (14 * 3600 + 30 * 60, 21 * 3600),
    "20260305": (14 * 3600 + 30 * 60, 21 * 3600),
    "20260306": (14 * 3600 + 30 * 60, 21 * 3600),
    "20260309": (13 * 3600 + 30 * 60, 20 * 3600),
    "20260310": (13 * 3600 + 30 * 60, 20 * 3600),
    "20260311": (13 * 3600 + 30 * 60, 20 * 3600),
    "20260312": (13 * 3600 + 30 * 60, 20 * 3600),
}

MAX_LAG_NS = max(LAGS_MS) * 1_000_000
MAX_H_NS = max(HORIZONS_S) * 1_000_000_000
TAIL_GUARD_NS = MAX_LAG_NS + MAX_H_NS + 1_000_000_000

# SPY MBO ingest
TICK_SIZE_FIXED = 10_000_000
INVALID_PRICE = 9_223_372_036_854_775_807

# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("es_spy_ic_v342")


# --------------------------------------------------------------------------
# Model loading -- CNN-Mamba v3.4.2 trunk (= CNNMambaV32 architecture)
# --------------------------------------------------------------------------
def load_es_model(device: str):
    """Load v3.4.2 fold_03 checkpoint into CNNMambaV32.

    The v3.4.2 ckpt at hc477fix_v2/fold_03_intra_ckpt.pt has NO book_cnn keys
    (verified: 227 keys, only t1/t2/t3 adapters+backbones + trunk + heads).
    The book pathway was gated to zero contribution, so the trunk-only model
    reproduces v3.4.2 inference exactly.
    """
    import torch
    from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import CNNMambaV32

    model = CNNMambaV32()
    ckpt = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)
    state = ckpt["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"v3.4.2 ckpt load failed: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model = model.to(device).eval()
    logger.info(
        f"v3.4.2 model loaded from {WEIGHTS_PATH.name} "
        f"(fold={ckpt.get('fold')}, epoch={ckpt.get('epoch')}, "
        f"n_heads={len(model.heads)})"
    )
    return model


def load_feature_stats() -> Dict[str, np.ndarray]:
    fs = np.load(str(STATS_PATH))
    out = {k: fs[k] for k in fs.files}
    logger.info(f"Loaded feature_stats keys={list(out.keys())}, mean_t1 shape={out['mean_t1'].shape}")
    return out


# --------------------------------------------------------------------------
# Per-date ES inference using v3.4.2 + SmartV32Dataset
# --------------------------------------------------------------------------
def es_inference_for_date(date_str: str, model, feat_stats, device: str) -> Dict[str, np.ndarray]:
    """Returns dict: ts_pred(ns), pred_1s, pred_5s, pred_10s, all from log_ret_*s heads.

    Uses SmartV32Dataset to construct T1/T2/T3 windows exactly as training did.
    Window-end timestamp == ts_pred (the event the model is predicting from).
    """
    import torch
    from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (
        SmartV32Dataset, WINDOW_SIZE_T1,
    )

    cache = LOCAL_CACHE_DIR / f"preds_v342_{date_str}.npz"
    if cache.exists():
        logger.info(f"[{date_str}] using cached v3.4.2 ES predictions: {cache}")
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    rth_start_sec, rth_end_sec = RTH_BY_DATE[date_str]

    logger.info(f"[{date_str}] building SmartV32Dataset...")
    t0 = time.time()
    ds = SmartV32Dataset(
        data_dir=ES_EVENTS_DIR,
        fifo_label_dir=FIFO_LABEL_DIR,
        alpha_label_dir=ALPHA_LABEL_DIR,
        pt_pred_dir=PT_PRED_DIR,
        dates=[date_str],
        tier2_parquet_root=TIER2_PARQUET,
        tier3_parquet_root=TIER3_PARQUET,
        feature_stats=feat_stats,
        cache_size=1,
        require_alpha_labels=False,
    )
    n_samples = len(ds)
    logger.info(f"[{date_str}] dataset built in {time.time()-t0:.1f}s, n_samples={n_samples}")
    if n_samples == 0:
        return {}

    # Recover per-sample window-end timestamps and RTH-filter
    day_data = ds._get_day(date_str)
    ts_arr = day_data["timestamps_ns"]
    sample_idx = np.array([(start, start + WINDOW_SIZE_T1 - 1)
                           for (_d, start, _wk) in ds.sample_index], dtype=np.int64)
    end_idx = sample_idx[:, 1]
    ts_pred_all = ts_arr[end_idx]

    # RTH window filter with tail guard
    sec_in_day = (ts_pred_all // 1_000_000_000) % 86400
    keep = (sec_in_day >= rth_start_sec) & (sec_in_day < (rth_end_sec - TAIL_GUARD_NS // 1_000_000_000))
    keep_indices = np.where(keep)[0]
    n_keep = len(keep_indices)
    logger.info(f"[{date_str}] {n_keep:,}/{n_samples:,} samples pass RTH+tail-guard filter")
    if n_keep == 0:
        return {}

    # Inference in batches -- pull samples one-at-a-time to avoid DataLoader overhead
    # (cache_size=1, single-day, sequential access -> fast)
    pred_1s = np.empty(n_keep, dtype=np.float32)
    pred_5s = np.empty(n_keep, dtype=np.float32)
    pred_10s = np.empty(n_keep, dtype=np.float32)
    ts_pred_out = ts_pred_all[keep_indices]

    t1_buf = []
    t2_buf = []
    t3_buf = []
    out_pos = 0
    t_inf = time.time()
    with torch.no_grad():
        for b0 in range(0, n_keep, BATCH_SIZE):
            b1 = min(b0 + BATCH_SIZE, n_keep)
            t1_buf.clear(); t2_buf.clear(); t3_buf.clear()
            for j in range(b0, b1):
                idx = int(keep_indices[j])
                events_dict, _targets, _masks = ds[idx]
                t1_buf.append(events_dict["events_t1"])
                t2_buf.append(events_dict["events_t2"])
                t3_buf.append(events_dict["events_t3"])
            batch = {
                "events_t1": torch.stack(t1_buf).to(device, non_blocking=True),
                "events_t2": torch.stack(t2_buf).to(device, non_blocking=True),
                "events_t3": torch.stack(t3_buf).to(device, non_blocking=True),
            }
            preds = model(batch)
            pred_1s[out_pos:out_pos + (b1 - b0)] = preds["log_ret_1s"].detach().cpu().numpy().astype(np.float32)
            pred_5s[out_pos:out_pos + (b1 - b0)] = preds["log_ret_5s"].detach().cpu().numpy().astype(np.float32)
            pred_10s[out_pos:out_pos + (b1 - b0)] = preds["log_ret_10s"].detach().cpu().numpy().astype(np.float32)
            out_pos += (b1 - b0)
            if b0 % (BATCH_SIZE * 20) == 0:
                logger.info(f"[{date_str}] inferred {b1}/{n_keep} ({(b1/n_keep)*100:.1f}%) "
                            f"elapsed={time.time()-t_inf:.1f}s")

    logger.info(f"[{date_str}] inference done in {time.time()-t_inf:.1f}s, "
                f"pred_5s mean={pred_5s.mean():.4f} std={pred_5s.std():.4f}")

    out = {
        "ts_pred": ts_pred_out,
        "pred_1s": pred_1s,
        "pred_5s": pred_5s,
        "pred_10s": pred_10s,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    logger.info(f"[{date_str}] cached v3.4.2 preds -> {cache}")
    return out


# --------------------------------------------------------------------------
# SPY mid extraction (unchanged from v2 script -- model-independent)
# --------------------------------------------------------------------------
def extract_spy_mid_for_date(date_str: str) -> Dict[str, np.ndarray]:
    cache = LOCAL_CACHE_DIR / f"spy_mid_{date_str}.npz"
    if cache.exists():
        logger.info(f"[{date_str}] using cached SPY mids: {cache}")
        d = np.load(cache)
        return {"ts": d["ts"], "mid": d["mid"]}

    import databento as db
    from sortedcontainers import SortedDict

    raw_path = SPY_RAW_DIR / f"xnas-itch-{date_str}.mbo.dbn.zst"
    logger.info(f"[{date_str}] decoding SPY DBN: {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    bid_levels = SortedDict()
    ask_levels = SortedDict()
    last_trade_price = 0
    rth_start_sec, rth_end_sec = RTH_BY_DATE[date_str]

    store = db.DBNStore.from_file(str(raw_path))
    ts_list, mid_list = [], []
    raw_total = 0
    t0 = time.time()
    LOG_INTERVAL = 2_000_000

    def refresh_best():
        nonlocal bid_levels, ask_levels
        while bid_levels and bid_levels.peekitem(-1)[1] <= 0:
            bid_levels.popitem(-1)
        while ask_levels and ask_levels.peekitem(0)[1] <= 0:
            ask_levels.popitem(0)
        bb = bid_levels.peekitem(-1)[0] if bid_levels else 0
        ba = ask_levels.peekitem(0)[0] if ask_levels else 0
        if bb > 0 and ba > 0 and ba >= bb:
            return (bb + ba) / 2.0
        return float(last_trade_price) if last_trade_price > 0 else 0.0

    for r in store:
        raw_total += 1
        if raw_total % LOG_INTERVAL == 0:
            logger.info(f"[{date_str}] SPY scanned {raw_total:,}")
        act = chr(r.action) if isinstance(r.action, int) else str(r.action)
        if act == 'R':
            bid_levels = SortedDict(); ask_levels = SortedDict(); last_trade_price = 0
            continue
        sid = chr(r.side) if isinstance(r.side, int) else str(r.side)
        price = int(r.price) if r.price != INVALID_PRICE else 0
        qty = int(r.size)
        if price <= 0:
            mid_list.append(0.0); ts_list.append(int(r.ts_event))
            continue
        if act == 'A':
            if sid == 'B': bid_levels[price] = bid_levels.get(price, 0) + qty
            elif sid == 'A': ask_levels[price] = ask_levels.get(price, 0) + qty
        elif act == 'C':
            if sid == 'B' and price in bid_levels: bid_levels[price] -= qty
            elif sid == 'A' and price in ask_levels: ask_levels[price] -= qty
        elif act == 'M':
            if sid == 'B': bid_levels[price] = bid_levels.get(price, 0) + qty
            elif sid == 'A': ask_levels[price] = ask_levels.get(price, 0) + qty
        elif act in ('T', 'F'):
            last_trade_price = price
            if sid == 'B' and price in ask_levels: ask_levels[price] -= qty
            elif sid == 'A' and price in bid_levels: bid_levels[price] -= qty
        else:
            mid_list.append(refresh_best()); ts_list.append(int(r.ts_event))
            continue
        mid = refresh_best()
        mid_list.append(mid); ts_list.append(int(r.ts_event))

    logger.info(f"[{date_str}] SPY decode done in {time.time()-t0:.1f}s, raw={raw_total:,}")
    ts_arr = np.array(ts_list, dtype=np.int64)
    mid_arr = np.array(mid_list, dtype=np.float64)

    last = 0.0
    for i in range(len(mid_arr)):
        if mid_arr[i] > 0: last = mid_arr[i]
        elif last > 0: mid_arr[i] = last

    sec_in_day = (ts_arr // 1_000_000_000) % 86400
    rth_mask = (sec_in_day >= rth_start_sec) & (sec_in_day < rth_end_sec) & (mid_arr > 0)
    ts_r = ts_arr[rth_mask]; mid_r = mid_arr[rth_mask]
    logger.info(f"[{date_str}] SPY RTH events: {len(ts_r):,}")

    mid_fixed = mid_r.astype(np.int64)
    if not np.all(np.diff(ts_r) >= 0):
        logger.warning(f"[{date_str}] SPY timestamps not monotonic -- sorting")
        order = np.argsort(ts_r); ts_r = ts_r[order]; mid_fixed = mid_fixed[order]

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ts=ts_r, mid=mid_fixed)
    logger.info(f"[{date_str}] cached SPY mids -> {cache}")
    return {"ts": ts_r, "mid": mid_fixed}


# --------------------------------------------------------------------------
# Rank IC
# --------------------------------------------------------------------------
def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float('nan')
    xr = _rankdata(x[m]); yr = _rankdata(y[m])
    xr -= xr.mean(); yr -= yr.mean()
    denom = float(np.sqrt((xr * xr).sum() * (yr * yr).sum()))
    if denom == 0: return float('nan')
    return float((xr * yr).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64) + 1.0
    sa = a[order]
    i = 0; n = len(sa)
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]: j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) * 0.5
            ranks[order[i:j]] = avg
        i = j
    return ranks


def build_records_for_date(date_str, preds, spy):
    ts_pred = preds["ts_pred"]
    ts_spy = spy["ts"]
    mid_spy = spy["mid"].astype(np.float64)
    out = {}
    pred_h_map = {1: preds["pred_1s"], 5: preds["pred_5s"], 10: preds["pred_10s"]}
    for lag_ms in LAGS_MS:
        lag_ns = lag_ms * 1_000_000
        idx_entry = np.searchsorted(ts_spy, ts_pred + lag_ns, side='left')
        entry_valid = idx_entry < len(ts_spy)
        idx_entry_c = np.clip(idx_entry, 0, len(ts_spy) - 1)
        mid_entry = mid_spy[idx_entry_c]
        for h_s in HORIZONS_S:
            target_ts = ts_pred + lag_ns + h_s * 1_000_000_000
            idx_exit = np.searchsorted(ts_spy, target_ts, side='left')
            exit_valid = idx_exit < len(ts_spy)
            idx_exit_c = np.clip(idx_exit, 0, len(ts_spy) - 1)
            mid_exit = mid_spy[idx_exit_c]
            valid = entry_valid & exit_valid & (mid_entry > 0) & (mid_exit > 0)
            fwd_ret = np.where(valid, (mid_exit - mid_entry) / mid_entry, np.nan)
            out[(lag_ms, h_s)] = {
                "pred": pred_h_map[h_s],
                "fwd_ret": fwd_ret.astype(np.float32),
                "valid": valid,
            }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    import torch
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"MODEL: v3.4.2 production champion (HC #529)")
    logger.info(f"WEIGHTS: {WEIGHTS_PATH}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000"))
        mlflow.set_experiment("spy_exec_only_lead_lag")
        mlflow_run = mlflow.start_run(run_name="es_spy_lead_lag_ic_v342")
        mlflow.log_params({
            "model_version": "v3.4.2",
            "weights_path": str(WEIGHTS_PATH),
            "window_t1": 1500, "stride": 250,
            "lags_ms": str(LAGS_MS), "horizons_s": str(HORIZONS_S),
            "n_dates": len(DATES),
            "dates": ",".join(DATES),
        })
        mlflow_enabled = True
    except Exception as e:
        logger.warning(f"MLflow disabled: {e}")
        mlflow_enabled = False
        mlflow_run = None

    model = load_es_model(device)
    feat_stats = load_feature_stats()

    per_day_ic = {}
    per_day_n = {}
    concat_buf = {(lag, h): {"pred": [], "fwd_ret": []} for lag in LAGS_MS for h in HORIZONS_S}

    smoke_only = os.environ.get("SMOKE_TEST", "0") == "1"
    dates_to_run = DATES[:1] if smoke_only else DATES
    if smoke_only:
        logger.info(f"SMOKE_TEST=1 -- running ONLY {dates_to_run[0]}")

    for d in dates_to_run:
        t_day = time.time()
        try:
            preds = es_inference_for_date(d, model, feat_stats, device)
        except Exception as e:
            logger.exception(f"[{d}] ES inference failed: {e}")
            continue
        if not preds:
            continue
        try:
            spy = extract_spy_mid_for_date(d)
        except Exception as e:
            logger.exception(f"[{d}] SPY extract failed: {e}")
            continue

        records = build_records_for_date(d, preds, spy)
        day_ic = {}; day_n = {}
        for (lag, h), r in records.items():
            pred = r["pred"]; ret = r["fwd_ret"]; valid = r["valid"]
            ic = rank_ic(pred[valid], ret[valid])
            key = f"lag{lag}_h{h}s"
            day_ic[key] = ic
            day_n[key] = int(valid.sum())
            concat_buf[(lag, h)]["pred"].append(pred[valid])
            concat_buf[(lag, h)]["fwd_ret"].append(ret[valid])
        per_day_ic[d] = day_ic
        per_day_n[d] = day_n
        logger.info(f"[{d}] IC summary: " +
                    ", ".join(f"{k}={v:.4f}" for k, v in day_ic.items()))
        if mlflow_enabled:
            for k, v in day_ic.items():
                if math.isfinite(v):
                    mlflow.log_metric(f"day_{d}_{k}", v)
        logger.info(f"[{d}] full pipeline took {time.time()-t_day:.1f}s")

    if smoke_only:
        logger.info(f"SMOKE_TEST complete. Per-day IC: {per_day_ic}")
        if mlflow_enabled: mlflow.end_run()
        return

    # Concat IC
    concat_ic = {}; concat_n = {}
    for (lag, h), buf in concat_buf.items():
        if not buf["pred"]:
            concat_ic[f"lag{lag}_h{h}s"] = float('nan'); concat_n[f"lag{lag}_h{h}s"] = 0
            continue
        p = np.concatenate(buf["pred"]); r = np.concatenate(buf["fwd_ret"])
        ic = rank_ic(p, r)
        concat_ic[f"lag{lag}_h{h}s"] = ic
        concat_n[f"lag{lag}_h{h}s"] = len(p)
        if mlflow_enabled and math.isfinite(ic):
            mlflow.log_metric(f"concat_{lag}_h{h}s", ic)

    finite = {k: v for k, v in concat_ic.items() if math.isfinite(v)}
    if finite:
        best_key = max(finite, key=lambda k: finite[k]); best_ic = finite[best_key]
    else:
        best_key = None; best_ic = float('nan')

    threshold = 0.05; min_days = 5
    decision_table = {}
    for key in concat_ic:
        pos_days = sum(1 for d in DATES
                       if d in per_day_ic
                       and per_day_ic[d].get(key) is not None
                       and math.isfinite(per_day_ic[d][key])
                       and per_day_ic[d][key] >= threshold)
        decision_table[key] = pos_days
    go_keys = [k for k, n in decision_table.items() if n >= min_days and concat_ic[k] >= threshold]
    decision = "GO" if go_keys else "NO-GO"

    matrix = {
        "model_version": "v3.4.2",
        "weights_path": str(WEIGHTS_PATH),
        "lags_ms": LAGS_MS, "horizons_s": HORIZONS_S,
        "concat_ic": concat_ic, "concat_n": concat_n,
        "per_day_ic": per_day_ic, "per_day_n": per_day_n,
        "best_key": best_key, "best_concat_ic": best_ic,
        "decision_threshold": threshold, "min_days_required": min_days,
        "days_passing_per_key": decision_table, "go_keys": go_keys,
        "decision": decision, "dates": DATES,
    }
    out_json = OUTPUT_DIR / "lag_ic_matrix.json"
    with open(out_json, "w") as f:
        json.dump(matrix, f, indent=2, default=str)
    logger.info(f"Saved {out_json}")
    if mlflow_enabled:
        try: mlflow.log_artifact(str(out_json))
        except Exception as e: logger.warning(f"mlflow.log_artifact failed: {e}")
        mlflow.log_metric("best_concat_ic", best_ic if math.isfinite(best_ic) else -999)
        mlflow.log_param("decision", decision)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mat = np.full((len(LAGS_MS), len(HORIZONS_S)), np.nan)
        for i, lag in enumerate(LAGS_MS):
            for j, h in enumerate(HORIZONS_S):
                mat[i, j] = concat_ic.get(f"lag{lag}_h{h}s", float('nan'))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
        ax.set_xticks(range(len(HORIZONS_S))); ax.set_xticklabels([f"{h}s" for h in HORIZONS_S])
        ax.set_yticks(range(len(LAGS_MS))); ax.set_yticklabels([f"{l}ms" for l in LAGS_MS])
        ax.set_xlabel("Forward horizon"); ax.set_ylabel("ES->SPY lag")
        ax.set_title(f"v3.4.2 ES->SPY concat rank IC (9d). Best={best_key}: {best_ic:.4f}")
        for i in range(len(LAGS_MS)):
            for j in range(len(HORIZONS_S)):
                v = mat[i, j]
                if math.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color=("white" if abs(v) > 0.05 else "black"), fontsize=9)
        plt.colorbar(im, ax=ax, label="Rank IC")
        plt.tight_layout()
        heatmap_path = OUTPUT_DIR / "lag_ic_heatmap.png"
        plt.savefig(heatmap_path, dpi=120); plt.close()
        logger.info(f"Saved {heatmap_path}")
        if mlflow_enabled:
            try: mlflow.log_artifact(str(heatmap_path))
            except Exception as e: logger.warning(f"mlflow.log_artifact heatmap failed: {e}")
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")

    lines = []
    lines.append("# ES->SPY Lead-Lag IC Study -- REPORT (v3.4.2)")
    lines.append("")
    lines.append("**Model**: CNN-Mamba **v3.4.2** (production champion per HC #529)  ")
    lines.append(f"**Weights**: `{WEIGHTS_PATH.name}` from `{WEIGHTS_PATH.parent.name}`  ")
    lines.append(f"**Window/stride**: 1500/250 events (T1) + T2 100ms-buckets + T3 1Hz  ")
    lines.append(f"**Primary directional head**: log_ret_5s (concat IC ~0.108-0.141 OOT)  ")
    lines.append(f"**Lags (ms)**: {LAGS_MS}  ")
    lines.append(f"**Horizons (s)**: {HORIZONS_S}  ")
    lines.append(f"**Days analyzed**: {len(per_day_ic)}/{len(DATES)} ({', '.join(per_day_ic.keys())})")
    lines.append("")
    lines.append(f"## Headline")
    lines.append(f"- **Decision**: {decision}")
    lines.append(f"- **Best (lag, horizon)**: `{best_key}` with concat IC = `{best_ic:.4f}`")
    lines.append(f"- **Threshold**: concat IC >= {threshold} AND >= {min_days} days positive at that threshold")
    lines.append("")
    lines.append("## Concat IC matrix (lag x horizon)")
    lines.append("")
    hdr = "| lag\\h | " + " | ".join(f"{h}s" for h in HORIZONS_S) + " |"
    sep = "|" + "---|" * (len(HORIZONS_S) + 1)
    lines.append(hdr); lines.append(sep)
    for lag in LAGS_MS:
        row = [f"{lag}ms"]
        for h in HORIZONS_S:
            v = concat_ic.get(f"lag{lag}_h{h}s", float('nan'))
            row.append(f"{v:+.4f}" if math.isfinite(v) else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Days passing threshold (count >= 0.05)")
    lines.append("")
    for k, n in sorted(decision_table.items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {n}/{len(per_day_ic)} days >= 0.05")
    lines.append("")
    lines.append("## Per-day IC at best (lag, horizon)")
    lines.append("")
    if best_key:
        lines.append("| date | IC | n samples |")
        lines.append("|---|---|---|")
        for d in DATES:
            ic = per_day_ic.get(d, {}).get(best_key, float('nan'))
            n = per_day_n.get(d, {}).get(best_key, 0)
            ic_str = f"{ic:+.4f}" if math.isfinite(ic) else "n/a"
            lines.append(f"| {d} | {ic_str} | {n:,} |")
    lines.append("")
    lines.append("## Why")
    if decision == "GO":
        lines.append(f"v3.4.2 signal `{best_key}` produced concat IC {best_ic:+.4f} "
                     f"on N={concat_n.get(best_key, 0):,} samples, with "
                     f"{decision_table.get(best_key, 0)}/{len(per_day_ic)} days at or above 0.05. "
                     f"Advance to passive-fill simulation.")
    else:
        lines.append(f"No (lag, horizon) cleared the 0.05 concat-IC AND >=5-days-positive bar. "
                     f"Best was `{best_key}` at {best_ic:+.4f}. NO-GO on SPY-execution-only at "
                     f"the lags/horizons tested.")
    lines.append("")
    lines.append("## Run notes")
    lines.append(f"- ES events: `{ES_EVENTS_DIR}`")
    lines.append(f"- SPY raw DBN: `{SPY_RAW_DIR}`")
    lines.append(f"- v3.4.2 weights: `{WEIGHTS_PATH}`")
    lines.append(f"- MLflow run: `{mlflow_run.info.run_id if mlflow_run else 'n/a'}`")
    lines.append("")

    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved {report_path}")
    if mlflow_enabled:
        try: mlflow.log_artifact(str(report_path))
        except Exception as e: logger.warning(f"mlflow.log_artifact report failed: {e}")
        mlflow.end_run()

    print(json.dumps({
        "decision": decision, "best_key": best_key,
        "best_concat_ic": best_ic, "days_analyzed": len(per_day_ic),
    }, indent=2))


if __name__ == "__main__":
    main()
