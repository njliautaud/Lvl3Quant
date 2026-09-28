#!/usr/bin/env python3
"""
ORB + Iceberg Exit Hybrid v2
============================
Uses book_tensors.npz directly — no pre-computed iceberg preds needed.
Computes iceberg signal inline from depth asymmetry at L1.

Strategy:
1. ORB: detect breakout of opening range (5/10/15 min)
2. Enter on breakout (limit at range edge, assume fill)
3. Exit when iceberg signal fires OR max hold expires
4. Iceberg signal: |depth_imb_L1| > cv_thresh (computed inline)

Data: /home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot/*.npz
Output: /home/jupiter/Lvl3Quant/data/processed/orb_iceberg_hybrid_v2_results.json
"""
import glob, json, os, sys, time, math
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

TICK = 0.25
TICK_VALUE = 12.50
RT_COST = 0.25  # ticks RT (limit entry at ORB = 0 slippage, exit = ~0.25 tick)
DATA = '/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot'
OUT = '/home/jupiter/Lvl3Quant/data/processed/orb_iceberg_hybrid_v2_results.json'
LOG = '/home/jupiter/Lvl3Quant/logs/orb_iceberg_hybrid_v2.log'
N_WORKERS = 8

# Sweep configs
ORB_BARS = [50, 100, 150]       # 5min=50bars, 10min=100bars, 15min=150bars (100ms bars)
MAX_HOLD_BARS = [300, 600, 1200]  # 30min/60min/120min
ICEBERG_CV_THRESH = [1.5, 2.0, 3.0, 5.0]  # |L1_imb_z| > thresh = iceberg signal
VOL_FILTER = [True, False]       # filter out high-vol days
ICEBERG_WINDOW = 50              # bars for iceberg z-score normalization


def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f'[{t}] {msg}'
    print(line, flush=True)
    try:
        with open(LOG, 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


def rolling_mean(a, w):
    cs = np.cumsum(a.astype(np.float64))
    out = np.empty(len(a), np.float64)
    out[:w] = cs[:w] / np.arange(1, w + 1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out.astype(np.float32)


def rolling_std(a, w):
    m = rolling_mean(a, w)
    m2 = rolling_mean(a ** 2, w)
    return np.sqrt(np.maximum(m2 - m ** 2, 0) + 1e-10).astype(np.float32)


def compute_iceberg_signal(book, window=50):
    """
    Iceberg signal: L1 depth imbalance z-score.
    High |z| when one side dominates at L1 = hidden iceberg order.
    Returns z-score array (positive = more bids, negative = more asks).
    """
    bid_depth = book[:, 0, 1].astype(np.float32)  # best bid depth
    ask_depth = book[:, 10, 1].astype(np.float32) # best ask depth
    imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-6)
    m = rolling_mean(imb, window)
    s = rolling_std(imb, window)
    return (imb - m) / (s + 1e-6)


def compute_vol_z(mid, window=1000):
    """Volatility z-score: >0 = high vol, <0 = low vol."""
    ret = np.diff(mid)
    rv = np.sqrt(rolling_mean(ret ** 2, window))
    m = rolling_mean(rv, window * 5)
    s = rolling_std(rv, window * 5)
    result = np.zeros(len(mid), dtype=np.float32)
    result[1:] = (rv - m) / (s + 1e-6)
    return result


def run_one_day(args):
    fpath, orb_bars, max_hold_bars, iceberg_thresh, vol_filter = args
    try:
        d = np.load(fpath, allow_pickle=False)
        book = d['book_tensors'].astype(np.float32)  # (N, 20, 4)
        mid = d['mid_prices'].astype(np.float32)
        N = len(mid)

        if N < orb_bars + max_hold_bars + 100:
            return None

        # Compute signals
        iceberg_z = compute_iceberg_signal(book, ICEBERG_WINDOW)
        vz = compute_vol_z(mid)

        # ORB: use first orb_bars as the opening range
        orh = float(np.max(mid[:orb_bars]))
        orl = float(np.min(mid[:orb_bars]))

        if orh - orl < TICK:
            return None  # flat open

        # Vol filter: skip high-vol days (vz at orb_bars > 0.5)
        if vol_filter and vz[orb_bars] > 0.5:
            return None

        # Find breakout entry
        entry_idx = None
        entry_dir = 0
        for i in range(orb_bars, min(orb_bars * 3, N - max_hold_bars - 10)):
            p = float(mid[i])
            if p > orh:
                entry_idx = i
                entry_dir = 1   # LONG
                break
            elif p < orl:
                entry_idx = i
                entry_dir = -1  # SHORT
                break

        if entry_idx is None:
            return {'traded': False, 'reason': 'no_breakout'}

        entry_price = float(mid[entry_idx])

        # Find exit: iceberg signal OR max hold
        exit_idx = entry_idx + max_hold_bars
        exit_type = 'timeout'

        for j in range(entry_idx + 10, min(entry_idx + max_hold_bars, N)):
            # Iceberg exit: signal in opposite direction to trade
            # (iceberg = support/resistance = reversal signal)
            iz_val = float(iceberg_z[j])
            if entry_dir == 1 and iz_val < -iceberg_thresh:
                # Long trade, iceberg on ask side = resistance = exit
                exit_idx = j
                exit_type = 'iceberg'
                break
            elif entry_dir == -1 and iz_val > iceberg_thresh:
                # Short trade, iceberg on bid side = support = exit
                exit_idx = j
                exit_type = 'iceberg'
                break

        exit_idx = min(exit_idx, N - 1)
        exit_price = float(mid[exit_idx])

        # PnL
        raw_ticks = (exit_price - entry_price) / TICK * entry_dir
        net_ticks = raw_ticks - RT_COST
        pnl_dollars = net_ticks * TICK_VALUE
        hold_bars = exit_idx - entry_idx
        win = 1 if net_ticks > 0 else 0

        return {
            'traded': True,
            'entry_dir': entry_dir,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'hold_bars': hold_bars,
            'raw_ticks': round(raw_ticks, 3),
            'net_ticks': round(net_ticks, 3),
            'pnl_dollars': round(pnl_dollars, 2),
            'win': win,
            'exit_type': exit_type,
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log('=== ORB + Iceberg Exit Hybrid v2 ===')
    log(f'Data: {DATA}')

    files = sorted(glob.glob(os.path.join(DATA, '*_book_tensors.npz')))
    log(f'Found {len(files)} files')
    if not files:
        log('ERROR: No files found')
        sys.exit(1)

    # Build config matrix
    configs = []
    for orb_b in ORB_BARS:
        for hold_b in MAX_HOLD_BARS:
            for cv in ICEBERG_CV_THRESH:
                for vf in VOL_FILTER:
                    configs.append((orb_b, hold_b, cv, vf))

    log(f'Config matrix: {len(configs)} configs x {len(files)} days')

    all_results = {}
    total_configs = len(configs)

    for ci, (orb_b, hold_b, cv, vf) in enumerate(configs):
        cfg_key = f'orb{orb_b//10}min_hold{hold_b//10}min_cv{cv}_vf{int(vf)}'
        log(f'[{ci+1}/{total_configs}] {cfg_key}')

        args = [(f, orb_b, hold_b, cv, vf) for f in files]

        with Pool(N_WORKERS) as pool:
            day_results = pool.map(run_one_day, args)

        trades = [r for r in day_results if r and r.get('traded')]
        n_days = len([r for r in day_results if r is not None])
        n_traded_days = len(trades)

        if not trades:
            all_results[cfg_key] = {
                'orb_bars': orb_b,
                'hold_bars': hold_b,
                'iceberg_cv': cv,
                'vol_filter': vf,
                'n_days': n_days,
                'n_traded_days': 0,
                'sortino': 0.0,
                'pnl_per_day': 0.0,
            }
            continue

        pnls = np.array([t['net_ticks'] for t in trades], dtype=np.float32)
        wr = float(np.mean([t['win'] for t in trades]))
        avg_pnl = float(np.mean(pnls))
        neg = pnls[pnls < 0]
        ds = float(np.std(neg)) if len(neg) > 3 else 1e-9
        sortino = avg_pnl / ds if ds > 0 else 0.0
        pnl_dollars = float(np.sum([t['pnl_dollars'] for t in trades]))
        iceberg_exits = sum(1 for t in trades if t.get('exit_type') == 'iceberg')
        avg_hold = float(np.mean([t['hold_bars'] for t in trades]))

        result = {
            'orb_bars': orb_b,
            'hold_bars': hold_b,
            'iceberg_cv': cv,
            'vol_filter': vf,
            'n_days': n_days,
            'n_traded_days': n_traded_days,
            'n_trades': len(trades),
            'win_rate': round(wr, 4),
            'avg_ticks': round(avg_pnl, 4),
            'pnl_dollars_total': round(pnl_dollars, 2),
            'pnl_dollars_per_day': round(pnl_dollars / max(n_days, 1), 2),
            'sortino': round(sortino, 4),
            'iceberg_exit_pct': round(iceberg_exits / max(len(trades), 1), 4),
            'avg_hold_bars': round(avg_hold, 1),
        }
        all_results[cfg_key] = result
        log(f'  n={len(trades)} WR={wr:.1%} ticks={avg_pnl:.3f} Sortino={sortino:.3f} $/day={pnl_dollars/max(n_days,1):.0f} ib_exit={iceberg_exits/max(len(trades),1):.0%}')

    # Save
    with open(OUT, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f'DONE. Results: {OUT}')

    # Summary
    ranked = sorted(all_results.items(), key=lambda x: x[1].get('sortino', -999), reverse=True)
    log('\n=== TOP 10 BY SORTINO ===')
    log(f'{"Config":<45} {"Sortino":>8} {"WR":>7} {"ticks":>8} {"$/day":>9} {"ib_exit%":>9}')
    for k, v in ranked[:10]:
        if v.get('n_traded_days', 0) < 5:
            continue
        log(f'{k:<45} {v.get("sortino",0):>8.3f} {v.get("win_rate",0):>7.1%} {v.get("avg_ticks",0):>8.3f} {v.get("pnl_dollars_per_day",0):>9.0f} {v.get("iceberg_exit_pct",0):>9.1%}')


if __name__ == '__main__':
    main()
