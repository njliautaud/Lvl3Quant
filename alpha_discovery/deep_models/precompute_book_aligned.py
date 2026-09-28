#!/usr/bin/env python3
"""
Precompute event-aligned book features to uncompressed .npy files so we can
mmap them at training time (the source mbo_book_normalized npz files are
DEFLATE-compressed → np.load mmap_mode='r' silently materializes the whole
array in RAM, OOMing 32GB Neptune).

Per date, writes to <CACHE>/<date>/{book_shape.npy, book_dyn.npy}
where rows are aligned 1:1 to the smart_v3 event index for that date.

Usage:
  python precompute_book_aligned.py --start 20260101 --end 20260309
"""
import argparse
import os
import socket
import sys
from pathlib import Path
import numpy as np

HOST = socket.gethostname().lower()
if "neptune" in HOST or os.path.exists("/home/nick/Lvl3Quant"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

EVENT_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
BOOK_DIR  = ROOT / "data" / "processed" / "mbo_book_normalized"
CACHE_DIR = ROOT / "data" / "processed" / "book_aligned_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def precompute_date(date_str, force=False):
    out_dir = CACHE_DIR / date_str
    bs_p = out_dir / "book_shape.npy"
    bd_p = out_dir / "book_dyn.npy"
    if bs_p.exists() and bd_p.exists() and not force:
        return f"skip-exists {date_str}"

    ev_p = EVENT_DIR / f"{date_str}_mbo_events.npz"
    bk_p = BOOK_DIR / f"{date_str}_book_norm.npz"
    if not ev_p.exists() or not bk_p.exists():
        return f"missing  {date_str}"

    ev = np.load(ev_p, mmap_mode="r")
    ev_ts = np.asarray(ev["timestamps"])
    bk = np.load(bk_p)  # decompresses fully, but only once
    bk_ts = bk["timestamps"]
    idx = np.searchsorted(bk_ts, ev_ts, side="right") - 1
    idx = np.clip(idx, 0, len(bk_ts) - 1)

    bs_aligned = np.asarray(bk["book_shape"][idx], dtype=np.float32)
    bd_aligned = np.asarray(bk["book_dynamics"][idx], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(bs_p, bs_aligned)
    np.save(bd_p, bd_aligned)
    return f"ok       {date_str} N={len(ev_ts):,} bs={bs_aligned.shape} bd={bd_aligned.shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260101")
    ap.add_argument("--end",   default="20260309")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = sorted(EVENT_DIR.glob("*_mbo_events.npz"))
    dates = [f.name[:8] for f in files if args.start <= f.name[:8] <= args.end]
    print(f"Precomputing {len(dates)} dates → {CACHE_DIR}")
    for d in dates:
        try:
            print(precompute_date(d, force=args.force), flush=True)
        except Exception as e:
            print(f"err      {d}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
