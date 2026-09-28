#!/usr/bin/env python3
"""
Queue-fade PASSIVE entry + iceberg-conditioned exit, low-vol gated.

Hypothesis:
  - Queue_fade signal in low-vol regime (IC=0.055, validated) → entry
  - Passive limit order at touch price → zero slippage, ~20-30% fill
  - Exit: time-based (hold 10s/30s/60s) with tight TP/SL
  - Low-vol gate: vol_50 < p33 (regime where IC doubles)

This tests the "patience" approach: wait for the signal + vol filter, then passively
enter. Lower trade count (~50-100/day) but higher win rate per the IC findings.

Complements Razer's qf_iceberg_chase (chase entry, all-vol) job running in parallel.

Signal computation from book_tensors:
  - queue_fade: L1 bid depletion rate > threshold AND sustained (anti-momentum of queue)
  - Actually: we re-use imb_z as proxy since it correlates with queue dynamics
  - True queue_fade = bid[t] / bid[t-k] < threshold (queue shrinking)

173 OOT days, parallel execution, 36 param configs.
Output: /home/jupiter/Lvl3Quant/data/processed/qf_passive_lowvol/
"""
import glob, json, os, subprocess, time, sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
BOOK_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
MBO_DIR   = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/qf_passive_lowvol")
PRED_DIR  = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Sweep configs: hold × TP × SL × vol_gate × signal_threshold
CONFIGS = []
for hold_ms in [10_000, 30_000, 60_000]:
    for tp, sl in [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]:
        for vol_pct in [33, 50]:    # low-vol gate: bottom 33% or 50%
            for qf_thresh in [1.5, 2.0, 2.5]:  # queue-fade z-score threshold
                CONFIGS.append({
                    'hold_ms': hold_ms,
                    'take_profit_ticks': tp,
                    'stop_loss_ticks': sl,
                    'vol_pct': vol_pct,
                    'qf_thresh': qf_thresh,
                    'name': f'hold{hold_ms//1000}s_tp{tp:.0f}_sl{sl:.0f}_vp{vol_pct}_qf{qf_thresh:.1f}',
                })

print(f"Total configs: {len(CONFIGS)}")

def rolling_mean(arr, w):
    cs = np.cumsum(arr.astype(np.float64))
    out = np.empty(len(arr), dtype=np.float32)
    out[:w] = cs[:w] / np.arange(1, w+1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out

def rolling_std(arr, w):
    rm = rolling_mean(arr, w)
    return np.sqrt(rolling_mean((arr - rm)**2, w) + 1e-10)

def compute_queue_fade_signal(book, mid, vol_pct, qf_thresh):
    """
    Queue-fade signal: L1 bid queue shrinking faster than expected.

    Signal logic:
    1. Compute L1 bid size at each bar: bid_qty = book[:,0,1]
    2. Compute L1 ask size: ask_qty = book[:,10,1]
    3. Queue imbalance: (bid - ask) / (bid + ask)
    4. Z-score this imbalance with rolling window
    5. Vol regime gate: compute rolling volatility, keep bottom vol_pct%

    Returns signal array: +1 (long), -1 (short), 0 (no signal)
    """
    bid = book[:, 0, 1].astype(np.float32)
    ask = book[:, 10, 1].astype(np.float32)
    total = bid + ask + 1e-6
    imb = (bid - ask) / total

    # Z-score (window=10 bars ≈ 10s for 1s bars, use 100 for std)
    imb_z = (imb - rolling_mean(imb, 10)) / (rolling_std(imb, 100) + 1e-6)

    # Queue fade: opposite direction from imbalance
    # When bid queue fading (large bid imbalance that's declining), SHORT signal
    # When ask queue fading (large ask imbalance declining), LONG signal
    # Proxy: use negative z-score cross as signal (mean-reversion of imbalance)
    qf_long  = imb_z < -qf_thresh   # ask pressure dominant → bounce up
    qf_short = imb_z >  qf_thresh   # bid pressure dominant → fade down

    # Vol regime gate
    rets = np.abs(np.diff(mid.astype(np.float32), prepend=mid[0]))
    vol = rolling_mean(rets, 50)
    vol_thresh = float(np.percentile(vol, vol_pct))
    low_vol = vol <= vol_thresh

    sig = np.zeros(len(mid), dtype=np.float32)
    sig[qf_long  & low_vol] =  1.0
    sig[qf_short & low_vol] = -1.0

    return sig, int(np.sum(sig > 0)), int(np.sum(sig < 0))


def process_day(args):
    """Generate signal and run fill_sim for one day."""
    book_file, cfg = args

    fname = os.path.basename(book_file)
    date = fname.replace("_book_tensors.npz", "")
    config_name = cfg['name']
    out_file = OUTPUT_DIR / f"{date}_{config_name}.json"

    if out_file.exists():
        return ('skip', f"{date}_{config_name}")

    # Generate signal
    pred_file = PRED_DIR / f"{date}_{config_name}_preds.npz"

    if not pred_file.exists():
        try:
            z = np.load(book_file, allow_pickle=False)
            book = z.get("book_tensors", z.get("book", None))
            mid  = z.get("mid_prices",  z.get("mid_price", z.get("mid", None)))
            if book is None or mid is None or book.ndim != 3:
                return ('skip_bad_data', f"{date}")
        except Exception as e:
            return ('error_load', f"{date}: {e}")

        sig, n_long, n_short = compute_queue_fade_signal(book, mid, cfg['vol_pct'], cfg['qf_thresh'])

        if n_long + n_short == 0:
            return ('skip_no_signal', f"{date}_{config_name}: no signals fired")

        # fill_sim_cli requires 'predictions' key
        np.savez_compressed(str(pred_file), predictions=sig, mid_prices=mid)

    # Find MBO file
    date_nodash = date.replace("-", "")
    mbo_glob = sorted(MBO_DIR.glob(f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"))
    if not mbo_glob:
        return ('skip_no_mbo', f"{date}")
    mbo_file = str(mbo_glob[0])

    # Run fill_sim_cli — use same CLI flags as imb_fillsim.py
    date_nodash = date.replace("-", "")
    mbo_file_path = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    if not mbo_file_path.exists():
        return ('skip_no_mbo', f"{date}")

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file_path),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", "0.5",
        "--latency-ms", "10",
        "--hold-ms", str(cfg['hold_ms']),
        "--take-profit-ticks", str(cfg['take_profit_ticks']),
        "--stop-loss-ticks", str(cfg['stop_loss_ticks']),
        # PASSIVE: no --chase-entry, no --market-entry flags
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode == 0:
            return ('ok', f"{date}_{config_name}")
        else:
            err = proc.stderr[-300:] if proc.stderr else proc.stdout[-300:]
            return ('error_sim', f"{date}_{config_name}: {err}")
    except subprocess.TimeoutExpired:
        return ('timeout', f"{date}_{config_name}")
    except Exception as e:
        return ('exception', f"{date}_{config_name}: {e}")


def aggregate():
    files = sorted(OUTPUT_DIR.glob("*.json"))
    if not files:
        print("No results to aggregate.")
        return

    from collections import defaultdict
    import statistics

    cfgs = defaultdict(lambda: {'pnl': [], 'trades': [], 'wr': [], 'fill_rate': []})

    for f in files:
        try:
            d = json.loads(f.read_text())
            # Config = everything after date prefix
            parts = f.stem.split('_', 3)
            if len(parts) < 4:
                continue
            cfg_key = parts[3]  # hold10s_tp2_sl4_vp33_qf1.5

            pnl = d.get('total_pnl_dollars', d.get('net_pnl_ticks', 0))
            n_tr = d.get('total_trades', d.get('n_trades', 0))
            wr   = d.get('win_rate', 0)
            fr   = d.get('fill_rate', 0)

            cfgs[cfg_key]['pnl'].append(pnl or 0)
            cfgs[cfg_key]['trades'].append(n_tr or 0)
            cfgs[cfg_key]['wr'].append(wr or 0)
            cfgs[cfg_key]['fill_rate'].append(fr or 0)
        except:
            pass

    ranked = []
    for k, v in cfgs.items():
        if not v['pnl']:
            continue
        n = len(v['pnl'])
        avg = sum(v['pnl']) / n
        std = statistics.stdev(v['pnl']) if n > 1 else 1.0
        neg = [p for p in v['pnl'] if p < 0]
        ds = statistics.stdev(neg) if len(neg) > 1 else (std or 1.0)
        sortino = avg / (ds + 1e-9)
        avg_tr = sum(v['trades']) / n
        avg_wr = sum(v['wr']) / n
        avg_fr = sum(v['fill_rate']) / n
        ranked.append((k, sortino, avg, n, avg_tr, avg_wr, avg_fr))

    ranked.sort(key=lambda x: -x[1])

    print(f"\n{'Config':<45} {'Sort':>7} {'Avg$/d':>9} {'Days':>5} {'Trd/d':>6} {'WR':>6} {'FillR':>6}")
    print('-' * 90)
    for cfg, s, a, n, t, wr, fr in ranked[:15]:
        sign = '+' if a >= 0 else ''
        print(f"{cfg:<45} {s:>7.3f} {sign}${a:>7.2f} {n:>5} {t:>6.1f} {wr:>5.1%} {fr:>5.1%}")

    pos = [(c, s, a) for c, s, a, *_ in ranked if s > 0 and a > 0]
    print(f"\nPositive (Sortino>0 AND PnL>0): {len(pos)}/{len(ranked)} configs")
    if pos:
        print(f"BEST: {pos[0][0]}  Sortino={pos[0][1]:.3f}  PnL=${pos[0][2]:.2f}/day")

    # Save summary
    summary = {
        'n_files': len(files),
        'n_configs': len(cfgs),
        'ranked': [
            {'config': c, 'sortino': s, 'avg_pnl_day': a, 'n_days': n,
             'avg_trades_day': t, 'avg_wr': wr, 'avg_fill_rate': fr}
            for c, s, a, n, t, wr, fr in ranked
        ]
    }
    summary_file = OUTPUT_DIR / 'qf_passive_lowvol_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved: {summary_file}")


def main():
    print('='*70)
    print('Queue-fade Passive Low-vol Fill Sim — Jupiter')
    print(f'Book cache: {BOOK_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print('='*70)

    book_files = sorted(BOOK_DIR.glob("*_book_tensors.npz"))
    print(f"Days available: {len(book_files)}")

    if not book_files:
        print("ERROR: No book_tensor files found!")
        sys.exit(1)

    # Check fill_sim binary
    if not Path(FILL_SIM).exists():
        print(f"WARNING: fill_sim not at {FILL_SIM}")
        # Try alternate paths
        for alt in [
            "/home/jupiter/Lvl3Quant/fill_sim/target/release/fill_sim_cli",
            "/home/jupiter/Lvl3Quant/rust_fill_sim/target/release/fill_sim_cli",
        ]:
            if Path(alt).exists():
                print(f"Found at: {alt}")
                break
        else:
            print("fill_sim binary not found — will report skip_no_sim")

    # Build work
    work = [(str(bf), cfg) for bf in book_files for cfg in CONFIGS]
    existing = len(list(OUTPUT_DIR.glob("*.json")))
    print(f"Total work units: {len(work)}, Already done: {existing}, Remaining: {len(work)-existing}")

    n_workers = min(multiprocessing.cpu_count(), 24)
    print(f"Workers: {n_workers}\n")

    stats = {}
    start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_day, w): w for w in work}
        for i, fut in enumerate(as_completed(futures)):
            status, msg = fut.result()
            stats[status] = stats.get(status, 0) + 1

            if i % 200 == 0:
                elapsed = time.time() - start
                done = sum(stats.values())
                rate = done / elapsed if elapsed > 0 else 0
                rem = (len(work) - done) / rate if rate > 0 else 0
                print(f"[{i+1}/{len(work)}] {stats} | rate={rate:.1f}/s ETA={rem/60:.0f}min")
            elif status.startswith('error') or status == 'exception':
                print(f"  [{status}] {msg[:80]}")

    print(f"\nFinal: {stats}")
    print(f"Elapsed: {(time.time()-start)/60:.1f} min\n")

    aggregate()


if __name__ == '__main__':
    main()
