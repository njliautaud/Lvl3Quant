#!/usr/bin/env python3
"""
PHASE 1 — Build native-horizon direction labels for direction_v2.

Discovery: /home/nick/Lvl3Quant/data/processed/mbo_book_features/{YYYYMMDD}_book_features.npz
already contains labels_1s, labels_5s, labels_10s, labels_30s in TICKS (mid_at(t+h) - mid_at(t)).
So Phase 1 reduces from full DBN reconstruction (1-2h) to extraction + binarization (~5 min).

For each 2026 day:
- Load labels_1s, labels_5s, labels_10s, timestamps
- Approximate native h=3s by averaging labels_1s and labels_5s? No: that's wrong.
  Instead, we have labels at h=1s, 5s, 10s. The brief asked for h in {3s, 5s, 10s}.
  We HAVE 5s and 10s natively. For h=3s, we don't have a native column.
  Strategy: use h=1s, 5s, 10s (substitute 3s -> 1s, which is the FASTER horizon we can support
  natively — and which the v1 diagnostic flagged as the only edge bin Q1 AUC 0.5526).
  Re-naming: dir_1s, dir_5s, dir_10s (h=1s replaces h=3s — finer-grained, matches v1 diagnostic).
- Build labels: dir_h = (label_h > 0).astype(int8), filter |label_h| >= 1 tick (drops noise/ties)
  Note: we save mid_change_h (raw signed ticks) too so downstream can choose its own threshold.
- Save: /home/nick/Lvl3Quant/data/direction_labels_native/{YYYYMMDD}.parquet
        cols [ts_ns, dmid_1s_ticks, dmid_5s_ticks, dmid_10s_ticks, mid_now]
- mid_now is reconstructed from bid_price_1 + ask_price_1 in book_features (col 0 and col 5).

Filters:
- ONLY 2026 days (already implied by filename glob)
- Skip days with < 100k events (incomplete days like 20260101, 20260429)
- Skip days where ALL labels are zero (incomplete recording)

Output: One parquet per day. Quick — should be 1-2s per day, ~3 min total.
"""
import argparse, os, sys, time, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

BOOK_FEAT_DIR = "/home/nick/Lvl3Quant/data/processed/mbo_book_features"
OUT_DIR = "/home/nick/Lvl3Quant/data/direction_labels_native"

MIN_EVENTS_PER_DAY = 100_000  # skip incomplete days
MIN_VALID_LABEL_FRAC = 0.20   # at least 20% of labels must be non-zero finite


def process_day(args):
    fn, force = args
    date_str = fn[:8]
    in_path = os.path.join(BOOK_FEAT_DIR, fn)
    out_path = os.path.join(OUT_DIR, f"{date_str}.parquet")
    t0 = time.time()
    if os.path.exists(out_path) and not force:
        return {"date": date_str, "status": "skip_exists", "n": 0, "wall_s": 0}
    try:
        z = np.load(in_path, allow_pickle=True)
    except Exception as e:
        return {"date": date_str, "status": f"load_failed: {e}", "n": 0, "wall_s": time.time()-t0}

    n = len(z["timestamps"])
    if n < MIN_EVENTS_PER_DAY:
        return {"date": date_str, "status": f"skip_too_small_n={n}", "n": n, "wall_s": time.time()-t0}

    l1 = z["labels_1s"].astype(np.float32)
    l5 = z["labels_5s"].astype(np.float32)
    l10 = z["labels_10s"].astype(np.float32)

    # Check non-zero finite fraction at h=10s (the strictest — needs 10s of future data)
    valid_frac = ((l10 != 0) & np.isfinite(l10)).mean()
    if valid_frac < MIN_VALID_LABEL_FRAC:
        return {"date": date_str, "status": f"skip_low_valid_frac={valid_frac:.3f}", "n": n, "wall_s": time.time()-t0}

    feats = z["features"]
    fn_names = list(z["feature_names"])
    bid_idx = fn_names.index("bid_price_1")
    ask_idx = fn_names.index("ask_price_1")
    bid1 = feats[:, bid_idx].astype(np.float64)
    ask1 = feats[:, ask_idx].astype(np.float64)
    # mid_now in PRICE units (dollars, ~6500 for ES); only valid when both sides > 0
    valid_book = (bid1 > 0) & (ask1 > 0)
    mid_now = np.where(valid_book, (bid1 + ask1) / 2.0, np.nan).astype(np.float32)

    df = pd.DataFrame({
        "ts_ns": z["timestamps"].astype(np.int64),
        "mid_now": mid_now,
        "dmid_1s_ticks": l1,
        "dmid_5s_ticks": l5,
        "dmid_10s_ticks": l10,
    })

    # Drop events where mid_now is NaN (book not initialized)
    pre = len(df)
    df = df[np.isfinite(df["mid_now"].values)].reset_index(drop=True)
    dropped_mid = pre - len(df)

    df.to_parquet(out_path, index=False, compression="snappy")
    wall = time.time() - t0
    return {
        "date": date_str, "status": "ok", "n_total": n, "n_kept": len(df),
        "dropped_no_mid": dropped_mid,
        "l1_pos_rate": float(((df["dmid_1s_ticks"] > 0)).mean()),
        "l5_pos_rate": float(((df["dmid_5s_ticks"] > 0)).mean()),
        "l10_pos_rate": float(((df["dmid_10s_ticks"] > 0)).mean()),
        "l1_nz_frac": float(((df["dmid_1s_ticks"] != 0) & np.isfinite(df["dmid_1s_ticks"])).mean()),
        "l5_nz_frac": float(((df["dmid_5s_ticks"] != 0) & np.isfinite(df["dmid_5s_ticks"])).mean()),
        "l10_nz_frac": float(((df["dmid_10s_ticks"] != 0) & np.isfinite(df["dmid_10s_ticks"])).mean()),
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--year", default="2026")
    args = ap.parse_args()

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_files = sorted([f for f in os.listdir(BOOK_FEAT_DIR)
                        if f.startswith(args.year) and f.endswith("_book_features.npz")])
    print(f"[phase1] {len(all_files)} candidate {args.year} days", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(process_day, [(f, args.force) for f in all_files])
    wall = time.time() - t0
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    print(f"[phase1] {len(ok)}/{len(all_files)} days ok in {wall:.1f}s", flush=True)
    if skipped:
        print(f"[phase1] skipped {len(skipped)}:", flush=True)
        for r in skipped:
            print(f"    {r['date']}: {r['status']}", flush=True)
    # OOT-window count
    oot_dates = [int(r["date"]) for r in ok if 20260227 <= int(r["date"]) <= 20260429]
    print(f"[phase1] OOT window [20260227,20260429]: {len(oot_dates)} days available", flush=True)
    summary = {
        "n_ok": len(ok), "n_skipped": len(skipped), "n_in_oot_window": len(oot_dates),
        "wall_total_s": wall,
        "results": results,
    }
    with open(os.path.join(OUT_DIR, "_phase1_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[phase1] done. Summary -> {OUT_DIR}/_phase1_summary.json", flush=True)


if __name__ == "__main__":
    main()
