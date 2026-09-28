"""
v3.4 data alignment check — verifies book features + MBO events are row-aligned by index
across all available dates, so the v3.4 Dataset class can join them via row index
(no timestamp join needed).

Output: scripts/v3_4_research/v34_alignment_manifest.json
  {
    "checked_dates": ["YYYYMMDD", ...],
    "aligned_dates": ["YYYYMMDD", ...],
    "misaligned_dates": [{"date": ..., "book_rows": N, "event_rows": M, "ts_match": bool}, ...],
    "book_feature_names": [...],  # 30 features from existing 5-level book
    "book_shape_reshape": "(N, 5_level_pairs, 4_features_per_pair) where 4=[bid_p, bid_s, ask_p, ask_s], 5=levels 1..5",
    "v3_4_oot_window_17day": "20260301..20260319",
    "v3_4_oot_dates_available": [...],
    "v3_4_train_window_60day": "20251207..20260227",
    "v3_4_train_dates_available": [...]
  }

Per HC #366: This is HC #307D-compliant new analysis tooling under scripts/v3_4_research/.
Does NOT modify any existing code.

Author: Claude (head-of-quant), 2026-05-14 21:55 ET, per user HC #366 autonomous execution.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_book_features")
EVENT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUTPUT = Path("/home/jupiter/Lvl3Quant/scripts/v3_4_research/v34_alignment_manifest.json")

# Falsification window per HC #365: 17-day OOT (March 2026)
OOT_17DAY = [f"202603{d:02d}" for d in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19]]
# Reasonable training window: ~60 trading days back from 20260227 (anchor before 5-day OOT)
TRAIN_60DAY_RANGE = ("20251207", "20260227")


def list_dates(d: Path, pattern: str) -> set:
    if not d.exists():
        return set()
    out = set()
    for f in d.glob(pattern):
        # pull leading YYYYMMDD
        stem = f.stem
        dt = stem.split("_")[0]
        if len(dt) == 8 and dt.isdigit():
            out.add(dt)
    return out


def main():
    book_dates = list_dates(BOOK_DIR, "*_book_features.npz")
    event_dates = list_dates(EVENT_DIR, "*_mbo_events.npz")
    common = sorted(book_dates & event_dates)
    print(f"[align-check] book dates: {len(book_dates)}, event dates: {len(event_dates)}, common: {len(common)}", flush=True)

    aligned = []
    misaligned = []
    book_feature_names = None
    book_shape_sample = None

    for i, date in enumerate(common):
        bp = BOOK_DIR / f"{date}_book_features.npz"
        ep = EVENT_DIR / f"{date}_mbo_events.npz"
        try:
            b = np.load(bp)
            e = np.load(ep)
        except Exception as exc:
            misaligned.append({"date": date, "error": f"load failed: {exc}"})
            continue

        b_rows = b["features"].shape[0]
        # event file uses "events" key per our inspection; rows match via len of timestamps
        ev_ts_key = "timestamps" if "timestamps" in e.files else ("ts_ns" if "ts_ns" in e.files else None)
        if ev_ts_key is None:
            misaligned.append({"date": date, "error": "no timestamps in event file"})
            continue
        e_rows = e[ev_ts_key].shape[0]

        # Quick alignment: count match + boundary timestamps match
        if b_rows != e_rows:
            misaligned.append({"date": date, "book_rows": int(b_rows), "event_rows": int(e_rows), "ts_match": False, "reason": "row count differs"})
            continue
        # Check first + last + middle timestamps
        try:
            b_ts = b["timestamps"]
            ts_match = (
                int(b_ts[0]) == int(e[ev_ts_key][0])
                and int(b_ts[-1]) == int(e[ev_ts_key][-1])
                and int(b_ts[b_rows // 2]) == int(e[ev_ts_key][e_rows // 2])
            )
        except Exception as exc:
            ts_match = False

        if ts_match:
            aligned.append(date)
        else:
            misaligned.append({"date": date, "book_rows": int(b_rows), "event_rows": int(e_rows), "ts_match": False, "reason": "timestamps differ"})

        # Capture feature names + sample shape once
        if book_feature_names is None and "feature_names" in b.files:
            book_feature_names = [str(x) for x in b["feature_names"]]
            book_shape_sample = list(b["features"].shape)

        if i % 20 == 0:
            print(f"[align-check] {i+1}/{len(common)} {date} -> aligned={len(aligned)} misaligned={len(misaligned)}", flush=True)

    # Subset checks for v3.4 windows
    aligned_set = set(aligned)
    oot_17d_avail = [d for d in OOT_17DAY if d in aligned_set]
    train_window = [d for d in sorted(aligned) if TRAIN_60DAY_RANGE[0] <= d <= TRAIN_60DAY_RANGE[1]]

    # Build reshape spec: 30 features -> 4 features per level pair x 5 levels + 10 derived (excluded from book trunk)
    # Map: indices 0-4 = bid_price_1..5, 5-9 = ask_price_1..5, 10-14 = bid_size_1..5, 15-19 = ask_size_1..5
    # Reshape into (5, 4) where features[level_pair, :] = [bid_price, bid_size, ask_price, ask_size]
    reshape_indices = {
        f"level_{lvl+1}": {
            "bid_price": lvl,        # bid_price_{lvl+1}
            "ask_price": 5 + lvl,    # ask_price_{lvl+1}
            "bid_size":  10 + lvl,   # bid_size_{lvl+1}
            "ask_size":  15 + lvl,   # ask_size_{lvl+1}
        }
        for lvl in range(5)
    }
    derived_indices = list(range(20, 30))  # cum_delta, rolling_imbalance, etc.

    manifest = {
        "schema_version": 1,
        "generated_at_et": "2026-05-14 21:55 ET",
        "purpose": "v3.4 dual-trunk Dataset class uses this manifest to know which dates are row-aligned book+event sources, and how to reshape the 30-flat book features into (5_level_pairs, 4_features) for the 2D-CNN book trunk.",
        "directives": ["HC #365 (v3.4 sole priority)", "HC #366 (lean-and-execute autonomous, Q1=A pick)", "HC #307D (analysis tooling)"],
        "checked_dates": common,
        "aligned_dates": aligned,
        "misaligned_dates": misaligned,
        "n_aligned": len(aligned),
        "n_misaligned": len(misaligned),
        "book_feature_names_30": book_feature_names,
        "book_features_sample_shape": book_shape_sample,
        "reshape_spec_5level_4feat": reshape_indices,
        "derived_indices_excluded_from_book_trunk": derived_indices,
        "v3_4_oot_17day_window": OOT_17DAY,
        "v3_4_oot_17day_available": oot_17d_avail,
        "v3_4_oot_17day_missing": [d for d in OOT_17DAY if d not in aligned_set],
        "v3_4_train_60day_range": TRAIN_60DAY_RANGE,
        "v3_4_train_60day_available": train_window,
        "v3_4_train_60day_count": len(train_window),
        "verdict": (
            "READY_FOR_TRAINER_WRITE" if (len(oot_17d_avail) >= 12 and len(train_window) >= 40)
            else "INSUFFICIENT_DATA"
        ),
        "verdict_explanation": (
            f"OOT 17-day window has {len(oot_17d_avail)}/15 aligned dates; "
            f"training 60-day window has {len(train_window)} aligned dates. "
            f"Need >=12 OOT + >=40 train for healthy fold-0 launch."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[align-check] wrote {OUTPUT}", flush=True)
    print(f"[align-check] VERDICT: {manifest['verdict']}", flush=True)
    print(f"[align-check] aligned={len(aligned)} misaligned={len(misaligned)} oot_17d_avail={len(oot_17d_avail)}/15 train_60d_avail={len(train_window)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
