#!/usr/bin/env python3
"""HC #441 — build per-fill CSV under the champion geometry
(SL=0.50, TP=3.00, hold=1.5s) for the short top-0.5% 1s band.

Reads the per-day caches built by hc437_pathB_exit_sweep.py and resolves each
fill analytically. Writes a single CSV with all fills across the 36 cached
dates. This CSV is the per-trade label set the Neptune confluence
meta-classifier will be trained on.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3 / "scripts/v3_4_research"))

# Resolver and constants come from hc437_pathB_exit_sweep
import hc437_pathB_exit_sweep as hc437  # noqa: E402

CACHE_DIR = LVL3 / "output/hc437_pathB_exit_sweep/fill_cache"
OUT_DIR = LVL3 / "output/hc441_champion_fills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Champion geometry candidates (HC #441 R1)
CONFIGS = [
    {"name": "R2_STRICT_AGGR", "SL": 0.50, "TP": 3.00, "H": 1.5},
    {"name": "R2_STRICT_CONS", "SL": 0.50, "TP": 2.50, "H": 1.5},
    {"name": "AGGR_LONGHOLD",  "SL": 0.50, "TP": 3.00, "H": 10.0},
]


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
    return {
        'fills': fills,
        'traj': trajs,
        'eod_ts': int(raw['eod_ts']),
    }


def main():
    dates = sorted([p.stem.replace('_cache', '')
                    for p in CACHE_DIR.glob("*_cache.npz")])
    print(f"Found {len(dates)} cached dates")

    for cfg in CONFIGS:
        SL, TP, H = cfg["SL"], cfg["TP"], cfg["H"]
        all_rows = []
        for d in dates:
            cache = load_cache(d)
            rows = hc437.resolve_cell(cache, TP, SL, H)
            all_rows.extend(rows)
        df = pd.DataFrame(all_rows)
        out_p = OUT_DIR / f"fills_{cfg['name']}_SL{SL}_TP{TP}_H{H}s.csv"
        df.to_csv(out_p, index=False)

        # Summary
        n = len(df)
        net = df['net_ticks'].mean() if n else float('nan')
        wr = (df['net_ticks'] > 0).mean() * 100 if n else float('nan')
        gross_w = df.loc[df['net_ticks'] > 0, 'net_ticks'].sum()
        gross_l = -df.loc[df['net_ticks'] <= 0, 'net_ticks'].sum()
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        sh = (df['net_ticks'].mean() / df['net_ticks'].std() * np.sqrt(n)
              if n and df['net_ticks'].std() > 0 else float('nan'))
        n_days = df['date'].nunique()
        per_day = df.groupby('date')['net_ticks'].sum()
        pos_days = (per_day > 0).sum()
        print(f"{cfg['name']:<18} SL={SL} TP={TP} H={H}s  "
              f"n={n} net={net:.4f} PF={pf:.3f} WR={wr:.1f}% "
              f"Sh√N={sh:.2f}  days={n_days} pos={pos_days}/{n_days}  "
              f"→ {out_p.name}")


if __name__ == "__main__":
    main()
