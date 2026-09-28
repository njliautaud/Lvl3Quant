#!/usr/bin/env python3
"""
ofi_features_v1.py — Build OFI features per event for OOT-overlap dates (HC #451 R3).

Question: Does Order Flow Imbalance (Cont/Kukanov/Stoikov-style) have standalone
predictive edge on realized MFE/MAE that the CNN-Mamba v3.4.2 baseline alpha
cannot extract on its own?

Inputs
------
- Raw mbo events: /home/jupiter/Lvl3Quant/data/processed/mbo_events/<YYYYMMDD>_mbo_events.npz
  Schema: events[:,0]=time_delta_log, [:,1]=event_type_id (0=A,1=C,2=M,3=T,4=F),
          [:,2]=side_id (0=Bid,1=Ask,2=None), [:,3]=price_rel_ticks,
          [:,4]=qty_log (size>=2 -> log>=ln(2)~0.693), [:,5]=spread_ticks
  timestamps: int64 ns since epoch

Features (per event, raw — NOT z-scored, so standalone edge test is interpretable)
-------------------------------------------------------------------------------
For window W seconds, summed strictly causally over the prior W seconds BEFORE
the event time (exclusive of current event):

  ofi_aggressive_<W>s:
    sum over prior W of: +size if (event_type==Trade AND side==Ask)  // buyer-init lift
                         -size if (event_type==Trade AND side==Bid)  // seller-init hit
    (Aggressive trade flow — buys minus sells.)
    Convention: Trade with side=Ask means the resting ask was lifted, i.e. buyer was aggressive.
                Trade with side=Bid means the resting bid was hit, i.e. seller was aggressive.

  trade_signed_flow_<W>s:
    Same as ofi_aggressive (alias for OFI-from-trades; kept for clarity).

  ofi_book_<W>s (Cont 2014 extension — book-event flow):
    sum over prior W of:
      bid-side ADD  -> +size  (bid grew = upward pressure)
      bid-side CXL  -> -size  (bid shrunk = downward)
      ask-side ADD  -> -size  (ask grew = downward)
      ask-side CXL  -> +size  (ask shrunk = upward)
    (Captures passive supply/demand imbalance.)

Provided at W in {1, 5, 10, 30} seconds. Per-event aligned to the raw mbo row count.

spread_ticks_now: events[:, 5]  (current spread at event time; raw)

queue_imbalance_bid: cannot be computed from this event stream (no live BB/BA sizes
in this NPZ schema). Logged as MISSING in feature_stats.

Outputs
-------
- Per-day NPZ at /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_ofi_features/<YYYYMMDD>_ofi.npz
- feature_stats.csv at /home/jupiter/Lvl3Quant/output/ofi_edge_v1/feature_stats.csv

NOTE on v3 events: although the smart_v3 corpus has columns 22-24 that are
z-scored OFI proxies, this script computes RAW OFI from raw mbo_events so that
buckets and TP/SL grids are interpretable in ticks/contracts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
RAW_MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUT_FEAT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_ofi_features")
OUT_STATS_DIR = Path("/home/jupiter/Lvl3Quant/output/ofi_edge_v1")
OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
V4_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v4")

OUT_FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STATS_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_SEC = [1, 5, 10, 30]
WALLTIME_CAP_SEC = 60 * 60

# Event/side codes (from process_missing_mbo.py)
ET_ADD, ET_CXL, ET_MOD, ET_TRADE, ET_FILL = 0, 1, 2, 3, 4
SIDE_BID, SIDE_ASK, SIDE_NONE = 0, 1, 2


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ----------------------------------------------------------------------------
def overlapping_dates() -> list[str]:
    oot = {p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz")}
    v4 = {p.name.split("_")[0] for p in V4_DIR.glob("*_alpha_labels.npz")}
    raw = {p.name.split("_")[0] for p in RAW_MBO_DIR.glob("*_mbo_events.npz")}
    return sorted(oot & v4 & raw)


def causal_window_sum(values: np.ndarray, ts_ns: np.ndarray, window_ns: int) -> np.ndarray:
    """For each event i, return sum of values[j] for all j with ts_ns[j] in [ts_ns[i] - W, ts_ns[i]).

    Strictly causal: excludes the current event from its own sum.
    Two-pointer over sorted timestamps. O(N).
    """
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    # cumulative sum trick: prefix[k] = sum of values[0..k-1]
    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(values.astype(np.float64))
    # for each i, find lo = smallest j s.t. ts_ns[j] >= ts_ns[i] - W
    # sum of [lo..i-1] = prefix[i] - prefix[lo]
    lower = ts_ns - window_ns
    # np.searchsorted for left bound
    lo = np.searchsorted(ts_ns, lower, side="left")
    out = prefix[np.arange(n)] - prefix[lo]
    return out


def build_features_for_date(date_str: str) -> dict | None:
    raw_path = RAW_MBO_DIR / f"{date_str}_mbo_events.npz"
    if not raw_path.exists():
        return None
    z = np.load(raw_path, allow_pickle=True)
    events = z["events"]
    ts = z["timestamps"].astype(np.int64)
    n = len(ts)

    et = events[:, 1].astype(np.int8)
    side = events[:, 2].astype(np.int8)
    qty_log = events[:, 4].astype(np.float32)
    spread_ticks = events[:, 5].astype(np.float32)

    # recover actual size: events[:,4] = log(max(size, 2)). Floor of 2 means
    # values below ln(2) ~ 0.693 should be clipped to 2; otherwise size = exp(qty_log).
    size = np.exp(qty_log.astype(np.float64))

    # Aggressive signed trade flow per event (only nonzero on Trade events):
    #   Trade + side=Ask -> aggressor=buyer -> +size
    #   Trade + side=Bid -> aggressor=seller -> -size
    trade_signed = np.zeros(n, dtype=np.float64)
    is_trade = (et == ET_TRADE)
    trade_signed[is_trade & (side == SIDE_ASK)] = size[is_trade & (side == SIDE_ASK)]
    trade_signed[is_trade & (side == SIDE_BID)] = -size[is_trade & (side == SIDE_BID)]

    # Book OFI per event (only nonzero on Add/Cancel events):
    #   bid ADD -> +size ; bid CXL -> -size
    #   ask ADD -> -size ; ask CXL -> +size
    book_signed = np.zeros(n, dtype=np.float64)
    is_add = (et == ET_ADD)
    is_cxl = (et == ET_CXL)
    book_signed[is_add & (side == SIDE_BID)] = size[is_add & (side == SIDE_BID)]
    book_signed[is_cxl & (side == SIDE_BID)] = -size[is_cxl & (side == SIDE_BID)]
    book_signed[is_add & (side == SIDE_ASK)] = -size[is_add & (side == SIDE_ASK)]
    book_signed[is_cxl & (side == SIDE_ASK)] = size[is_cxl & (side == SIDE_ASK)]

    save_dict = {}
    feature_summary = []
    for W in WINDOWS_SEC:
        wns = int(W * 1e9)
        ofi_trade = causal_window_sum(trade_signed, ts, wns).astype(np.float32)
        ofi_book = causal_window_sum(book_signed, ts, wns).astype(np.float32)
        save_dict[f"ofi_aggressive_{W}s"] = ofi_trade
        save_dict[f"trade_signed_flow_{W}s"] = ofi_trade  # alias
        save_dict[f"ofi_book_{W}s"] = ofi_book

        for name, arr in [(f"ofi_aggressive_{W}s", ofi_trade),
                          (f"ofi_book_{W}s", ofi_book)]:
            nan_frac = float(np.mean(~np.isfinite(arr)))
            zero_frac = float(np.mean(arr == 0))
            feature_summary.append(dict(
                date=date_str, feature=name, n=int(n),
                nan_frac=nan_frac, zero_frac=zero_frac,
                mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)),
                p05=float(np.nanpercentile(arr, 5)),
                p50=float(np.nanpercentile(arr, 50)),
                p95=float(np.nanpercentile(arr, 95)),
                abs_p95=float(np.nanpercentile(np.abs(arr), 95)),
            ))

    save_dict["spread_ticks_now"] = spread_ticks
    feature_summary.append(dict(
        date=date_str, feature="spread_ticks_now", n=int(n),
        nan_frac=float(np.mean(~np.isfinite(spread_ticks))),
        zero_frac=float(np.mean(spread_ticks == 0)),
        mean=float(np.nanmean(spread_ticks)), std=float(np.nanstd(spread_ticks)),
        p05=float(np.nanpercentile(spread_ticks, 5)),
        p50=float(np.nanpercentile(spread_ticks, 50)),
        p95=float(np.nanpercentile(spread_ticks, 95)),
        abs_p95=float(np.nanpercentile(np.abs(spread_ticks), 95)),
    ))

    # queue_imbalance_bid: not available — log MISSING
    feature_summary.append(dict(
        date=date_str, feature="queue_imbalance_bid", n=int(n),
        nan_frac=1.0, zero_frac=0.0, mean=np.nan, std=np.nan,
        p05=np.nan, p50=np.nan, p95=np.nan, abs_p95=np.nan,
    ))

    out_path = OUT_FEAT_DIR / f"{date_str}_ofi.npz"
    np.savez(out_path, **save_dict)
    return {"date": date_str, "n": n, "out": str(out_path), "summary": feature_summary}


def main():
    t0 = time.time()
    log(f"[start] ofi_features_v1")
    dates = overlapping_dates()
    log(f"[dates] overlap (oot ∩ v4 ∩ raw_mbo) = {len(dates)} dates: {dates[:3]}...{dates[-3:]}")

    all_summary = []
    built = []
    for d in dates:
        if time.time() - t0 > WALLTIME_CAP_SEC:
            log(f"[walltime] cap hit; built {len(built)}/{len(dates)}")
            break
        try:
            res = build_features_for_date(d)
            if res is None:
                log(f"  SKIP {d}: no raw file")
                continue
            all_summary.extend(res["summary"])
            built.append(d)
            log(f"  built {d}: N={res['n']:,} -> {Path(res['out']).name}")
        except Exception as e:
            import traceback
            log(f"  FAILED {d}: {e}\n{traceback.format_exc()}")

    df = pd.DataFrame(all_summary)
    csv_path = OUT_STATS_DIR / "feature_stats.csv"
    df.to_csv(csv_path, index=False)
    log(f"[write] {csv_path} ({len(df)} rows)")

    regen = {
        "task": "ofi_features_v1",
        "hc_refs": ["HC#451R3", "HC#485R5", "HC#420"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_dates_built": len(built),
        "windows_seconds": WINDOWS_SEC,
        "missing_features": ["queue_imbalance_bid (no live BB/BA size in event-stream)"],
        "elapsed_seconds": round(time.time() - t0, 1),
        "out_dir": str(OUT_FEAT_DIR),
        "stats_csv": str(csv_path),
    }
    sentinel = OUT_FEAT_DIR / ".regen_complete.json"
    with open(sentinel, "w") as f:
        json.dump(regen, f, indent=2)
    log(f"[done] elapsed {time.time()-t0:.1f}s, built {len(built)} dates")


if __name__ == "__main__":
    main()
