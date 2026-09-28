#!/usr/bin/env python3
"""
ofi_extend_20_dates.py — Generate OFI features for 20 dates missing from stacked confluence.
Extends coverage from 28 → 48 OOT dates for full HC #428 R1 regime-agnostic gate (≥40 days).
"""
import sys
import time
sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
from ofi_features_v1 import build_features_for_date, log, OUT_FEAT_DIR

MISSING_DATES = [
    "20260320", "20260322", "20260323", "20260324", "20260325",
    "20260326", "20260327", "20260329", "20260330", "20260331",
    "20260415", "20260416", "20260417", "20260419", "20260420",
    "20260422", "20260424", "20260426", "20260427", "20260429",
]

def main():
    t0 = time.time()
    log(f"[start] OFI extension for {len(MISSING_DATES)} missing dates")
    built = []
    for d in MISSING_DATES:
        out_path = OUT_FEAT_DIR / f"{d}_ofi.npz"
        if out_path.exists():
            log(f"  SKIP {d}: already exists")
            built.append(d)
            continue
        try:
            res = build_features_for_date(d)
            if res is None:
                log(f"  SKIP {d}: no raw MBO file")
                continue
            built.append(d)
            log(f"  BUILT {d}: N={res['n']:,}")
        except Exception as e:
            log(f"  FAILED {d}: {e}")
    elapsed = time.time() - t0
    log(f"[done] built {len(built)}/{len(MISSING_DATES)} dates in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
