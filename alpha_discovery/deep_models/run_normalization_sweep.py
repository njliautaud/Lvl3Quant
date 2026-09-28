#!/usr/bin/env python3
"""
Normalization + Rolling Mean Conviction Sweep
==============================================
Tests alternative signal normalizations:
1. Rolling mean z-score (smoothed conviction — 10, 50, 100 bar windows)
2. No vol gate (raw signal, no vol filtering)
3. Extreme vol gates (90th, 95th percentile)
4. Quantile normalization
5. EMA z-score (exponential moving average instead of expanding)
6. Rank normalization

Each normalization generates per-day NPZ files, then runs through
the Rust MBO fill sim with various hold times and conv thresholds.

Usage:
    python alpha_discovery/deep_models/run_normalization_sweep.py --workers 6
"""

import sys, json, time, bisect, logging, argparse, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_norm_sweep_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_norm_sweep_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('norm_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'norm_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Signal Processing Variants ──

def compute_trailing_vol(mid, window=3000):
    ret_1s = np.zeros(len(mid))
    ret_1s[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 10000
    vol = np.full(len(mid), np.nan)
    cs = np.cumsum(ret_1s)
    cs2 = np.cumsum(ret_1s ** 2)
    for i in range(window, len(mid)):
        s = cs[i] - cs[i - window]
        s2 = cs2[i] - cs2[i - window]
        m = s / window
        vol[i] = np.sqrt(max(s2 / window - m * m, 0))
    return vol


def precompute_vol_percentiles(vol, pcts):
    n = len(vol)
    result = {p: np.full(n, -np.inf) for p in pcts}
    sv = []
    for i in range(n):
        if not np.isnan(vol[i]):
            bisect.insort(sv, vol[i])
        if len(sv) >= 100:
            for p in pcts:
                result[p][i] = sv[min(int(len(sv) * p / 100), len(sv) - 1)]
    return result


def zscore_expanding(arr):
    result = np.full_like(arr, np.nan, dtype=np.float64)
    rs, rsq, c = 0.0, 0.0, 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v): continue
        rs += v; rsq += v * v; c += 1
        if c >= 50:
            m = rs / c
            result[i] = (v - m) / max(np.sqrt(rsq / c - m * m), 1e-8)
    return result


def zscore_ema(arr, span=3000):
    """EMA-based z-score — more responsive to recent data than expanding."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    ema_mean = 0.0
    ema_var = 0.0
    c = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v): continue
        c += 1
        if c == 1:
            ema_mean = v
            ema_var = 0.0
            continue
        ema_mean = alpha * v + (1 - alpha) * ema_mean
        ema_var = alpha * (v - ema_mean) ** 2 + (1 - alpha) * ema_var
        if c >= 50:
            std = max(np.sqrt(ema_var), 1e-8)
            result[i] = (v - ema_mean) / std
    return result


def zscore_rolling(arr, window=3000):
    """Fixed rolling window z-score."""
    import pandas as pd
    s = pd.Series(arr)
    rm = s.rolling(window, min_periods=50).mean()
    rs = s.rolling(window, min_periods=50).std().clip(lower=1e-8)
    return ((s - rm) / rs).values


def rolling_mean_signal(z_scores, window=50):
    """Smooth z-scores with a rolling mean for less noisy conviction."""
    import pandas as pd
    smoothed = pd.Series(z_scores).rolling(window, min_periods=1).mean().values
    return smoothed


def rank_normalize(arr, window=3000):
    """Rolling rank normalization — maps to [-1, 1] based on rank in recent window."""
    import pandas as pd
    s = pd.Series(arr)
    rank = s.rolling(window, min_periods=100).rank(pct=True).values
    # Map [0, 1] to [-1, 1] and scale to z-score-like range
    result = (rank * 2 - 1) * 3  # range [-3, 3]
    return np.nan_to_num(result, nan=0.0)


def time_mask(n_bars):
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 360)


# ── Normalization Configs ──
# Each generates a signal array from raw predictions + mid prices
# Format: (name, function(preds_aligned, mid) -> signal, vol_gates_to_test)

def make_normalizations():
    configs = []

    # 1. Baseline expanding z-score (what we currently use)
    def norm_expanding(preds, mid):
        return zscore_expanding(preds)
    configs.append(("expanding_zscore", norm_expanding, [0, 50, 70, 80]))

    # 2. EMA z-score (more responsive)
    for span in [1000, 3000, 5000]:
        def norm_ema(preds, mid, s=span):
            return zscore_ema(preds, span=s)
        configs.append((f"ema_zscore_span{span}", norm_ema, [0, 50, 70]))

    # 3. Rolling window z-score
    for w in [1000, 3000, 5000]:
        def norm_roll(preds, mid, win=w):
            return zscore_rolling(preds, window=win)
        configs.append((f"rolling_zscore_w{w}", norm_roll, [0, 50, 70]))

    # 4. Rolling mean smoothed conviction (THE USER'S IDEA)
    for smooth in [10, 50, 100, 200]:
        def norm_smooth(preds, mid, s=smooth):
            z = zscore_expanding(preds)
            return rolling_mean_signal(z, window=s)
        configs.append((f"smooth{smooth}_expanding", norm_smooth, [0, 50, 70]))

    # 5. Rank normalization
    configs.append(("rank_norm", lambda p, m: rank_normalize(p), [0, 50, 70]))

    # 6. Raw predictions (no normalization at all)
    def norm_raw(preds, mid):
        return preds
    configs.append(("raw_no_norm", norm_raw, [0]))

    return configs


# ── Sim configs per normalization ──
SIM_CONFIGS = []
for conv in [1.5, 2.0, 2.5, 3.0]:
    for hold_min in [10, 15, 20, 30]:
        hold_ms = hold_min * 60 * 1000
        SIM_CONFIGS.append((conv, hold_ms, True, f'conv{int(conv*10)}_hold{hold_min}m_chase'))
# Also test with take-profit
for conv in [2.0, 2.5]:
    for tp in [8, 10, 15]:
        SIM_CONFIGS.append((conv, 1800000, True, f'conv{int(conv*10)}_hold30m_tp{tp}_chase', tp))

# Pure TP/SL exits — no time limit (use very long hold as proxy)
# Stay in until either TP or SL hits
LONG_HOLD = 3600000  # 60 min max (effectively "no time exit")
for conv in [2.0, 2.5, 3.0]:
    for tp in [5, 8, 10, 15, 20, 30]:
        for sl in [10, 15, 20, 25]:
            SIM_CONFIGS.append((conv, LONG_HOLD, True,
                               f'conv{int(conv*10)}_tpsl_tp{tp}_sl{sl}_chase', tp, sl))

# Also test TP-only (no SL, long hold — let winners run)
for conv in [2.0, 2.5]:
    for tp in [5, 8, 10, 15, 20]:
        SIM_CONFIGS.append((conv, LONG_HOLD, True,
                           f'conv{int(conv*10)}_tponly_tp{tp}_chase', tp, None))


def run_sim(mbo_file, pred_file, output_file, conv, hold_ms, chase, tp=None, trail=None):
    cmd = [str(BINARY), '--mbo-file', str(mbo_file), '--predictions', str(pred_file),
           '--output', str(output_file), '--hold-ms', str(hold_ms),
           '--signal-threshold', str(conv), '--latency-ms', '0', '--quiet']
    if chase:
        cmd += ['--chase-entry', '--chase-max-ticks', '1', '--chase-max-reprices', '3']
    if tp is not None:
        cmd += ['--take-profit-ticks', str(tp)]
    if trail is not None:
        cmd += ['--trailing-ticks', str(trail)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    log.info("Loading WF predictions...")
    wf_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    normalizations = make_normalizations()
    log.info(f"Normalizations: {len(normalizations)}")
    log.info(f"Sim configs per normalization: {len(SIM_CONFIGS)}")

    # Step 1: Generate all prediction files
    log.info("Generating prediction files for each normalization...")
    saved = {}  # (date, norm_name, vg) -> filepath

    for norm_name, norm_fn, vol_gates in normalizations:
        for date in dates:
            preds_raw = wf_data[f'{date}_preds']
            mid = wf_data[f'{date}_mid']
            n = len(preds_raw)
            if n < 5000: continue

            # Align CNN offset
            aligned = np.zeros(n, dtype=np.float64)
            end = min(n, len(preds_raw) + CNN_OFFSET)
            aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

            # Apply normalization
            signal = norm_fn(aligned, mid)
            signal = np.nan_to_num(signal, nan=0.0)

            # Time mask
            tmask = time_mask(n)

            # Vol processing
            vol = compute_trailing_vol(mid)
            vol_thresholds = precompute_vol_percentiles(vol, tuple(vg for vg in vol_gates if vg > 0))

            for vg in vol_gates:
                sig = signal.copy()
                if vg > 0:
                    for i in range(len(sig)):
                        if np.isnan(vol[i]) or vol[i] < vol_thresholds.get(vg, np.full(n, -np.inf))[i]:
                            sig[i] = 0.0
                sig[~tmask] = 0.0

                fname = f'{date}_{norm_name}_vol{vg}.npz'
                fpath = PRED_OUT_DIR / fname
                np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
                saved[(date, norm_name, vg)] = str(fpath)

    log.info(f"Generated {len(saved)} prediction files")

    # Step 2: Run sims
    jobs = []
    for (date, norm_name, vg), pred_file in saved.items():
        date_compact = date.replace('-', '')
        mbo_candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn.zst'))
        if not mbo_candidates: continue
        mbo = mbo_candidates[0]

        for cfg in SIM_CONFIGS:
            tp, trail = None, None
            if len(cfg) == 6:
                conv, hold_ms, chase, label, tp, trail = cfg
            elif len(cfg) == 5:
                conv, hold_ms, chase, label, tp = cfg
            else:
                conv, hold_ms, chase, label = cfg
            out = SIM_OUT_DIR / f'{norm_name}_vol{vg}_{label}_{date}.json'
            if out.exists(): continue
            jobs.append((str(mbo), pred_file, str(out), conv, hold_ms, chase, tp, trail,
                        norm_name, vg, label, date))

    log.info(f"Jobs: {len(jobs)}")

    completed = 0
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for mbo, pred, out, conv, hold, chase, tp, trail, nn, vg, lbl, dt in jobs:
            f = executor.submit(run_sim, mbo, pred, out, conv, hold, chase, tp, trail)
            futures[f] = (nn, vg, lbl, dt)

        for future in as_completed(futures):
            nn, vg, lbl, dt = futures[future]
            completed += 1
            res = future.result()
            if res:
                results.append({'norm': nn, 'vg': vg, 'config': lbl, 'date': dt,
                               'pnl': res.get('total_pnl_dollars', 0),
                               'trades': res.get('total_trades', 0),
                               'signals': res.get('total_signals', 0),
                               'filled': res.get('total_filled', 0),
                               'wr': res.get('win_rate', 0)})
            if completed % 200 == 0:
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta:.1f}min)")

    log.info(f"Done: {completed} jobs in {time.time()-t0:.0f}s")

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r['norm'], r['vg'], r['config'])].append(r)

    summaries = []
    for (nn, vg, cfg), days in agg.items():
        tp = sum(d['pnl'] for d in days)
        tt = sum(d['trades'] for d in days)
        nd = len(days)
        dp = [d['pnl'] for d in days]
        avg = np.mean(dp)
        std = np.std(dp) if nd > 1 else 1
        sr = avg / std * np.sqrt(252) if std > 0 else 0
        wr = sum(d['wr'] * d['trades'] for d in days) / max(tt, 1)
        summaries.append({'norm': nn, 'vg': vg, 'config': cfg, 'pnl': round(tp, 2),
                         'trades': tt, 'sharpe': round(sr, 3), 'wr': round(wr, 4),
                         'annual': round(avg * 252, 0), 'n_days': nd})

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    log.info(f"\n{'='*100}")
    log.info("NORMALIZATION SWEEP RESULTS — Top 50 by Sharpe")
    log.info(f"{'='*100}")
    for i, s in enumerate(summaries[:50]):
        log.info(f"#{i+1:>3} {s['norm']:<30} vol{s['vg']:<3} {s['config']:<35} "
                f"Sharpe {s['sharpe']:>7.2f} P&L ${s['pnl']:>10,.2f} {s['trades']:>4}t "
                f"WR {s['wr']*100:>5.1f}% Ann ${s['annual']:>9,.0f}")

    with open(RESULTS_DIR / f'norm_sweep_results_{_ts}.json', 'w') as f:
        json.dump({'summaries': summaries}, f, indent=2)
    log.info("Saved.")


if __name__ == '__main__':
    main()
