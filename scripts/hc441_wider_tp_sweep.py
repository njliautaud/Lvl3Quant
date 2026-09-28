#!/usr/bin/env python3
"""HC #441 robustness check — extend TP sweep beyond 3.0 ticks to find
the diminishing-returns boundary. Uses cached trajectories so this
finishes in seconds. Train/test split: first 21 chronological dates train,
last 15 test (matches HC #441 holdout)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3 / "scripts/v3_4_research"))
import hc437_pathB_exit_sweep as hc437  # noqa: E402

CACHE_DIR = LVL3 / "output/hc437_pathB_exit_sweep/fill_cache"
OUT_DIR = LVL3 / "output/hc441_champion_fills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wider TP grid, anchored at champion's SL=0.5 and R2-strict H=1.5
SLs = [0.50, 0.57]
TPs = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
Hs  = [1.0, 1.5, 2.0, 5.0, 10.0]


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
    return {'fills': fills, 'traj': trajs, 'eod_ts': int(raw['eod_ts'])}


def metrics(df):
    if len(df) == 0:
        return None
    net = df['net_ticks']
    mu, sd = net.mean(), net.std()
    sh = mu / sd * np.sqrt(len(df)) if sd > 0 else float('nan')
    gw = net[net > 0].sum()
    gl = -net[net <= 0].sum()
    pf = gw / gl if gl > 0 else float('inf')
    wr = (net > 0).mean() * 100
    per_day = df.groupby('date')['net_ticks'].sum()
    pos = (per_day > 0).sum()
    return dict(n=len(df), net=mu, PF=pf, WR=wr, Sh=sh,
                n_days=per_day.shape[0], pos_days=pos)


def main():
    dates = sorted([p.stem.replace('_cache', '')
                    for p in CACHE_DIR.glob("*_cache.npz")])
    train_dates = set(dates[:21])

    rows = []
    for SL in SLs:
        for TP in TPs:
            for H in Hs:
                all_rows = []
                for d in dates:
                    cache = load_cache(d)
                    all_rows.extend(hc437.resolve_cell(cache, TP, SL, H))
                df = pd.DataFrame(all_rows)
                tr = df[df['date'].isin(train_dates)]
                te = df[~df['date'].isin(train_dates)]
                m_all = metrics(df)
                m_tr = metrics(tr)
                m_te = metrics(te)
                rows.append({'SL': SL, 'TP': TP, 'H': H,
                             'ALL_n': m_all['n'],   'ALL_net': m_all['net'],
                             'ALL_PF': m_all['PF'], 'ALL_Sh':  m_all['Sh'],
                             'TRAIN_net': m_tr['net'], 'TRAIN_Sh': m_tr['Sh'],
                             'TEST_n': m_te['n'],      'TEST_net': m_te['net'],
                             'TEST_PF': m_te['PF'],    'TEST_Sh': m_te['Sh'],
                             'TEST_posdays': m_te['pos_days'],
                             'TEST_ndays': m_te['n_days']})
                print(f"SL={SL} TP={TP} H={H:>4}s  "
                      f"TRAIN net={m_tr['net']:+.3f} Sh={m_tr['Sh']:>6.2f}  "
                      f"TEST n={m_te['n']} net={m_te['net']:+.3f} "
                      f"PF={m_te['PF']:.2f} Sh={m_te['Sh']:>6.2f} "
                      f"pos={m_te['pos_days']}/{m_te['n_days']}")
    out_df = pd.DataFrame(rows)
    out_p = OUT_DIR / "wider_tp_sweep.csv"
    out_df.to_csv(out_p, index=False)
    print(f"\nWrote {out_p}")

    # Sort by TEST Sharpe (out-of-sample robustness)
    print("\n=== TOP-10 by TEST Sharpe ===")
    top = out_df.sort_values('TEST_Sh', ascending=False).head(10)
    print(top[['SL', 'TP', 'H', 'TRAIN_net', 'TRAIN_Sh',
               'TEST_n', 'TEST_net', 'TEST_PF', 'TEST_Sh',
               'TEST_posdays', 'TEST_ndays']].to_string(index=False))


if __name__ == "__main__":
    main()
