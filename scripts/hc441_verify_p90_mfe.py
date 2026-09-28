#!/usr/bin/env python3
"""HC #441 R2 compliance check: compute p90 realized MFE within h=1s
for the short top-0.5% 1s band, restricted to the training slice
(first 21 chronological cached dates). If p90_MFE@1s < 3.00 ticks the
champion downgrades to CONSERV (TP=2.50)."""
import sys
from pathlib import Path
import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3 / "scripts/v3_4_research"))

CACHE_DIR = LVL3 / "output/hc437_pathB_exit_sweep/fill_cache"
TICK_RAW = 25_000_000  # 0.25 ES points in raw price units (from hc437 module)


def load_cache(date_str: str) -> dict:
    p = CACHE_DIR / f"{date_str}_cache.npz"
    raw = np.load(p, allow_pickle=True)
    fills = list(raw['fills'])
    lens = raw['traj_lens']
    flat = raw['traj_flat']
    trajs = []
    cursor = 0
    for L in lens:
        trajs.append(flat[cursor:cursor + L])
        cursor += L
    return {'fills': fills, 'traj': trajs}


def main():
    dates = sorted([p.stem.replace('_cache', '')
                    for p in CACHE_DIR.glob("*_cache.npz")])
    n_train = 21  # train slice = first 21 chronological dates
    train_dates = dates[:n_train]
    test_dates  = dates[n_train:]
    print(f"Train dates: {len(train_dates)} ({train_dates[0]}..{train_dates[-1]})")
    print(f"Test  dates: {len(test_dates)} ({test_dates[0]}..{test_dates[-1]})")

    for h_s in (1.0, 1.5, 2.0, 5.0, 10.0):
        hold_ns = int(h_s * 1e9)
        for label, ds in [("TRAIN", train_dates), ("TEST", test_dates),
                           ("ALL", dates)]:
            mfes = []  # in ticks
            for d in ds:
                cache = load_cache(d)
                for fill, traj in zip(cache['fills'], cache['traj']):
                    if traj.size == 0:
                        mfes.append(0.0)
                        continue
                    offs = traj[:, 0]
                    prs = traj[:, 1]
                    within = offs <= hold_ns
                    if not within.any():
                        mfes.append(0.0); continue
                    prs_h = prs[within]
                    entry = fill['entry_price_raw']
                    direction = fill['direction']
                    if direction == 'long':
                        max_gain_raw = (prs_h.max() - entry)
                    else:
                        max_gain_raw = (entry - prs_h.min())
                    mfes.append(max_gain_raw / TICK_RAW)
            arr = np.array(mfes)
            p50, p75, p90, p95, p99 = np.percentile(arr, [50, 75, 90, 95, 99])
            print(f"  h={h_s:>4}s {label:<5} n={len(arr)} "
                  f"mean={arr.mean():.3f} p50={p50:.2f} p75={p75:.2f} "
                  f"p90={p90:.2f} p95={p95:.2f} p99={p99:.2f}")


if __name__ == "__main__":
    main()
