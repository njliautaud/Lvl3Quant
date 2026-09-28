#!/usr/bin/env python3
"""HC #439 — Build per-signal dataframe with predictions, ts_ns, MFE/MAE per
horizon, and all filter features (vol/volume/tod/horizon-confluence/spread/
book imbalance/trade aggression/day-of-week).

Per HC #439 task spec:
- Predictions NPZ source: cnn_mamba_v2_bulk_oot_v2 (W=1000, stride=250)
- 48 OOT days (Feb 24 - Apr 29 2026)
- Output one parquet per day at output/hc439_deep_mfe_mae/signals/<DATE>.parquet

For each signal we compute:
- pred_1s, pred_5s, pred_10s
- label_1s, label_5s, label_10s (realized log-ret in ticks)
- mfe_1s_tk, mfe_5s_tk, mfe_10s_tk (signed for side; computed long-side here,
  flip sign for short)
- mae_1s_tk, mae_5s_tk, mae_10s_tk
- filter_vol_500ev_tk (rolling realized vol of mid-price last 500 trades in ticks)
- filter_evt_per_sec_30s (trade events per second last 30s)
- filter_tod_min  (RTH minute-of-day, 9:30=570, 16:00=960)
- filter_tod_bucket (e.g. '09:30-10:00', 'pre-open', 'post-close')
- filter_spread_tk (most recent trade-print spread proxy; we'll use rolling
  range/N as a noisy spread proxy in absence of bid/ask data)
- filter_book_imb (placeholder — no MBO L1 here, see note)
- filter_buy_aggr_50 (fraction of last 50 trades that uplifted px; proxy for
  trade aggression)
- filter_dow (0=Mon .. 4=Fri)

Pipeline outputs are pure numerical arrays for fast Phase 1-4 work.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = LVL3 / "data/derived/mid_price_cache_hc439"
PRED_DIR = LVL3 / "output/cnn_mamba_v2_bulk_oot_v2"
EV_DIR = LVL3 / "data/processed/mbo_events_smart_v3"  # for timestamps mapping
OUT_DIR = LVL3 / "output/hc439_deep_mfe_mae/signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000
STRIDE = 250
TICK_RAW = 250_000_000  # databento fixed point, 1 tick = 0.25 pts

ES_PX_RAW_MIN = 5000 * 1_000_000_000   # 5000 pts
ES_PX_RAW_MAX = 8000 * 1_000_000_000   # 8000 pts


def load_trades(date_str: str):
    """Load cached ES trade prints, filter to ES outright by price range."""
    p = CACHE_DIR / f"{date_str}_trades.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    ts = d['ts_ns'].astype(np.int64)
    px = d['price_raw'].astype(np.int64)
    mask = (px > ES_PX_RAW_MIN) & (px < ES_PX_RAW_MAX)
    return ts[mask], px[mask]


def load_event_timestamps(date_str: str):
    """Load smart_v3 mbo_events timestamps (per-event, used to map sample idx → ts_ns)."""
    p = EV_DIR / f"{date_str}_mbo_events.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return d['timestamps'].astype(np.int64)


def compute_mfe_mae(trades_ts: np.ndarray, trades_px: np.ndarray,
                    sig_ts: np.ndarray, horizons_ns: list[int]):
    """
    For each signal, compute MFE/MAE over forward horizons.
    Returns dict horizon_ns -> (mfe_raw, mae_raw) arrays of shape (N_signals,).
    MFE/MAE are computed long-side (max - entry, entry - min). Flip sign for short later.
    Values returned in raw databento units (price_raw).
    Entry price = trade at first ts >= sig_ts.
    """
    n = len(sig_ts)
    n_tr = len(trades_ts)
    result = {}
    # Pre-compute insertion points
    start_idx = np.searchsorted(trades_ts, sig_ts, side='left')
    # entry price = next trade at or after sig_ts
    entry_px = np.where(start_idx < n_tr, trades_px[np.clip(start_idx, 0, n_tr-1)], np.int64(0))
    for h_ns in horizons_ns:
        end_ts = sig_ts + h_ns
        end_idx = np.searchsorted(trades_ts, end_ts, side='right')
        mfe = np.zeros(n, dtype=np.float64)
        mae = np.zeros(n, dtype=np.float64)
        # Loop over signals (slow but vectorized inside)
        for i in range(n):
            s, e = start_idx[i], end_idx[i]
            if e <= s + 1:
                # No future ticks in horizon
                mfe[i] = 0.0
                mae[i] = 0.0
                continue
            seg = trades_px[s:e]
            mx = seg.max()
            mn = seg.min()
            ep = entry_px[i]
            # in ticks
            mfe[i] = (mx - ep) / TICK_RAW   # long-side favorable
            mae[i] = (ep - mn) / TICK_RAW   # long-side adverse
        result[h_ns] = (mfe, mae)
    return result, start_idx, entry_px


def compute_filters(trades_ts: np.ndarray, trades_px: np.ndarray,
                    sig_ts: np.ndarray, start_idx: np.ndarray):
    """Compute filter features per signal."""
    n = len(sig_ts)
    n_tr = len(trades_ts)
    # Realized vol (in ticks) over last 500 trades — std of price changes
    # Cumulative tick changes; use rolling window
    # Simpler: for each signal, take trades[start_idx-500:start_idx] and compute std of diffs
    vol_500 = np.zeros(n, dtype=np.float64)
    spread_proxy = np.zeros(n, dtype=np.float64)  # rolling tick-range/sqrt(N) proxy
    buy_aggr_50 = np.zeros(n, dtype=np.float64)
    evt_per_sec_30s = np.zeros(n, dtype=np.float64)

    H_30S = 30 * 1_000_000_000

    for i in range(n):
        s = start_idx[i]
        # last 500 trade prints
        a = max(0, s - 500)
        b = s
        if b - a > 5:
            seg = trades_px[a:b].astype(np.float64) / TICK_RAW   # in ticks
            diffs = np.diff(seg)
            vol_500[i] = diffs.std()
            spread_proxy[i] = (seg.max() - seg.min()) / max(1, int(np.sqrt(b - a)))
        # buy aggression from last 50 trades (uptick fraction)
        a2 = max(0, s - 50)
        b2 = s
        if b2 - a2 > 5:
            seg2 = trades_px[a2:b2]
            ups = (np.diff(seg2) > 0).sum()
            dns = (np.diff(seg2) < 0).sum()
            tot = ups + dns
            buy_aggr_50[i] = ups / tot if tot > 0 else 0.5
        # Events per second last 30s
        sig_t = sig_ts[i]
        a3 = np.searchsorted(trades_ts, sig_t - H_30S, side='left')
        b3 = start_idx[i]
        evt_per_sec_30s[i] = (b3 - a3) / 30.0
    return {
        "filter_vol_500ev_tk": vol_500,
        "filter_spread_proxy_tk": spread_proxy,
        "filter_buy_aggr_50": buy_aggr_50,
        "filter_evt_per_sec_30s": evt_per_sec_30s,
    }


def tod_bucket_from_ts_ns(ts_ns: int) -> tuple[str, float]:
    """Return (bucket_name, minutes_from_midnight_ET). ts_ns is in UTC.
    ET = UTC - 4 (EDT, May; -5 EST else). For Feb-Apr 2026: DST starts Mar 8,
    so Feb 24 - Mar 7 is EST (-5), Mar 8 onwards EDT (-4).
    Approximate by date.
    """
    # Convert ts_ns to UTC datetime
    sec = ts_ns / 1e9
    # Naive: minutes since midnight UTC
    utc_min = (sec % 86400) / 60
    # ET offset: simplified — for the range 2026-03-08+ use -240, else -300
    # We don't have date here, will compute outside in vectorized fashion.
    return None, utc_min


def tod_minutes_et(sig_ts: np.ndarray, date_str: str) -> np.ndarray:
    """Vector: minutes-of-day in ET for each signal."""
    # ET offset
    yyyymmdd = int(date_str)
    dst_start = 20260308  # second Sun in Mar 2026 = Mar 8
    dst_end = 20261101    # first Sun in Nov 2026
    if dst_start <= yyyymmdd < dst_end:
        offset_min = -240  # EDT
    else:
        offset_min = -300  # EST
    sec = sig_ts.astype(np.float64) / 1e9
    utc_min = (sec % 86400) / 60
    et_min = (utc_min + offset_min) % (24 * 60)
    return et_min


def tod_bucket_name(et_min: float) -> str:
    """Map ET minute-of-day to a session bucket."""
    if et_min < 8 * 60:
        return "overnight"
    if et_min < 9 * 60 + 30:
        return "pre-open"
    if et_min < 10 * 60:
        return "0930-1000"
    if et_min < 10 * 60 + 30:
        return "1000-1030"
    if et_min < 11 * 60:
        return "1030-1100"
    if et_min < 11 * 60 + 30:
        return "1100-1130"
    if et_min < 12 * 60:
        return "1130-1200"
    if et_min < 12 * 60 + 30:
        return "1200-1230"
    if et_min < 13 * 60:
        return "1230-1300"
    if et_min < 13 * 60 + 30:
        return "1300-1330"
    if et_min < 14 * 60:
        return "1330-1400"
    if et_min < 14 * 60 + 30:
        return "1400-1430"
    if et_min < 15 * 60:
        return "1430-1500"
    if et_min < 15 * 60 + 30:
        return "1500-1530"
    if et_min < 16 * 60:
        return "1530-1600"
    if et_min < 17 * 60:
        return "post-close"
    return "evening"


def day_of_week(date_str: str) -> int:
    """0=Mon..6=Sun."""
    from datetime import date
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    return date(y, m, d).weekday()


def process_one_day(date_str: str) -> dict:
    """Build per-signal feature table for one date. Returns dict of arrays + saves parquet."""
    t0 = time.time()
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    if not pred_path.exists():
        return {"date": date_str, "status": "no_pred"}
    pd_data = np.load(pred_path, allow_pickle=False)
    preds = pd_data['predictions'].astype(np.float64)  # (N, 3)
    labels = pd_data['labels'].astype(np.float64)      # (N, 3) in ticks already
    n_windows = int(pd_data['n_windows'])
    # Mapping sample_idx → event_idx → ts_ns
    ts_events = load_event_timestamps(date_str)
    if ts_events is None:
        return {"date": date_str, "status": "no_events"}
    n_events = len(ts_events)
    # sample i corresponds to event at min(i*250 + 1000 - 1, n-1)
    sample_idx = np.arange(n_windows)
    event_idx = np.minimum(sample_idx * STRIDE + WINDOW - 1, n_events - 1)
    sig_ts = ts_events[event_idx]
    # Only keep where labels are not NaN for at least one horizon
    valid = np.isfinite(labels).any(axis=1)
    if not valid.any():
        return {"date": date_str, "status": "no_valid_labels"}
    n_valid = int(valid.sum())
    preds = preds[valid]
    labels = labels[valid]
    sig_ts = sig_ts[valid]

    # Load trades
    trades_ts, trades_px = load_trades(date_str)
    if trades_ts is None or len(trades_ts) < 1000:
        return {"date": date_str, "status": "no_trades"}

    # Compute MFE/MAE at 1s/5s/10s
    horizons_s = [1.0, 5.0, 10.0]
    horizons_ns = [int(h * 1e9) for h in horizons_s]
    mfe_mae, start_idx, entry_px = compute_mfe_mae(trades_ts, trades_px, sig_ts, horizons_ns)

    # Filters
    filters = compute_filters(trades_ts, trades_px, sig_ts, start_idx)

    # TOD
    tod_min = tod_minutes_et(sig_ts, date_str)
    tod_buckets = np.array([tod_bucket_name(m) for m in tod_min], dtype=object)
    dow = day_of_week(date_str)

    out = {
        "date": np.array([date_str] * n_valid, dtype=object),
        "sig_ts_ns": sig_ts.astype(np.int64),
        "entry_px_raw": entry_px.astype(np.int64),
        "pred_1s": preds[:, 0].astype(np.float32),
        "pred_5s": preds[:, 1].astype(np.float32),
        "pred_10s": preds[:, 2].astype(np.float32),
        "label_1s": labels[:, 0].astype(np.float32),
        "label_5s": labels[:, 1].astype(np.float32),
        "label_10s": labels[:, 2].astype(np.float32),
        "mfe_1s_tk": mfe_mae[horizons_ns[0]][0].astype(np.float32),
        "mae_1s_tk": mfe_mae[horizons_ns[0]][1].astype(np.float32),
        "mfe_5s_tk": mfe_mae[horizons_ns[1]][0].astype(np.float32),
        "mae_5s_tk": mfe_mae[horizons_ns[1]][1].astype(np.float32),
        "mfe_10s_tk": mfe_mae[horizons_ns[2]][0].astype(np.float32),
        "mae_10s_tk": mfe_mae[horizons_ns[2]][1].astype(np.float32),
        "tod_min_et": tod_min.astype(np.float32),
        "tod_bucket": tod_buckets,
        "dow": np.array([dow] * n_valid, dtype=np.int8),
        **{k: v.astype(np.float32) for k, v in filters.items()},
    }
    # Save parquet
    import pandas as pd
    df = pd.DataFrame(out)
    out_path = OUT_DIR / f"{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    return {"date": date_str, "status": "ok", "n_signals": n_valid,
            "n_trades": int(len(trades_ts)), "elapsed_s": round(elapsed, 1),
            "out_path": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", help="explicit date list")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.dates:
        dates = args.dates
    else:
        dates = sorted([f.stem.replace('_predictions', '') for f in PRED_DIR.glob('*.npz')])

    print(f"Processing {len(dates)} dates with {args.workers} workers", flush=True)
    if args.workers == 1:
        for d in dates:
            print(process_one_day(d), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one_day, d): d for d in dates}
            for fut in as_completed(futs):
                try:
                    print(fut.result(), flush=True)
                except Exception as e:
                    import traceback
                    print({"date": futs[fut], "error": str(e), "tb": traceback.format_exc()[-500:]}, flush=True)


if __name__ == "__main__":
    main()
