#!/usr/bin/env python3
"""
Conviction Decay Exit Sweep
============================
Tests a DYNAMIC EXIT strategy encoded directly into prediction files.

Concept: Instead of fixed hold time, the signal stays non-zero ONLY while
the rolling mean z-score stays above an exit_threshold. When conviction
decays below exit_threshold, signal zeros out -> Rust sim exits at next bar.

Pipeline per bar:
  1. CNN offset alignment (19 bars)
  2. Expanding z-score (standard normalization)
  3. Rolling mean smoothing (window N)
  4. Vol gate + time mask
  5. DECAY EXIT LOGIC:
     - Entry: smoothed signal crosses ABOVE entry_threshold
     - Stay active while smoothed signal remains ABOVE exit_threshold
     - Exit: smoothed signal drops below exit_threshold -> zero signal
     - Re-entry: must cross entry_threshold again (hysteresis)

The Rust sim sees a signal that is non-zero only during "active conviction"
windows, with a 60-min safety-net max hold.

Sweep parameters:
  - Smoothing windows: 10, 50, 100 bars
  - Entry thresholds: 1.5, 2.0, 2.5
  - Exit thresholds: 0.0, 0.5, 1.0
  - Vol gates: 0, 50, 70
  - Conv thresholds (Rust sim): 0.1 (low, since signal is pre-filtered)
  - Max hold: 60 min (safety net)
  - Chase: 1t/3r (standard)

Usage:
    python alpha_discovery/deep_models/run_decay_exit_sweep.py --workers 24
"""

import sys, json, time, logging, argparse, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_decay_exit_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_decay_exit_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000  # 60 min safety net

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('decay_exit_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'decay_exit_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Signal Processing (optimized) ──

def compute_trailing_vol(mid, window=3000):
    """5-minute trailing volatility in bps. Vectorized."""
    n = len(mid)
    ret_1s = np.zeros(n)
    ret_1s[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1e-10) * 10000
    vol = np.full(n, np.nan)
    cs = np.cumsum(ret_1s)
    cs2 = np.cumsum(ret_1s ** 2)
    # Vectorized rolling window
    idx = np.arange(window, n)
    s = cs[idx] - cs[idx - window]
    s2 = cs2[idx] - cs2[idx - window]
    m = s / window
    vol[window:] = np.sqrt(np.maximum(s2 / window - m * m, 0))
    return vol


def compute_expanding_vol_percentile(vol, pct):
    """Compute expanding percentile for vol gating. Returns threshold array."""
    import pandas as pd
    s = pd.Series(vol)
    # expanding percentile — use rank approach
    result = np.full(len(vol), -np.inf)
    # Use expanding with quantile
    exp = s.expanding(min_periods=100)
    result_s = exp.quantile(pct / 100.0)
    result = result_s.values
    return result


def zscore_expanding_fast(arr):
    """Expanding z-score — vectorized where possible."""
    n = len(arr)
    result = np.full(n, 0.0, dtype=np.float64)
    # Use cumulative sums for mean and variance
    valid = ~np.isnan(arr)
    vals = np.where(valid, arr, 0.0)
    cs = np.cumsum(vals)
    cs2 = np.cumsum(vals ** 2)
    cc = np.cumsum(valid.astype(np.float64))

    # Only compute where count >= 50
    mask = cc >= 50
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return result

    counts = cc[idx]
    means = cs[idx] / counts
    vars_ = cs2[idx] / counts - means * means
    stds = np.sqrt(np.maximum(vars_, 0))
    stds = np.maximum(stds, 1e-8)
    result[idx] = (vals[idx] - means) / stds
    # Zero out where original was NaN
    result[~valid] = 0.0
    return result


def rolling_mean_smooth(z_scores, window):
    """Smooth z-scores with rolling mean for conviction signal."""
    import pandas as pd
    return pd.Series(z_scores).rolling(window, min_periods=1).mean().values


def time_mask(n_bars):
    """Skip first 30 min and last 30 min of session."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 360)


# ── Conviction Decay Exit Logic ──

def _apply_decay_exit_python(smoothed_signal, entry_thresh, exit_thresh):
    """Pure Python fallback for decay exit."""
    n = len(smoothed_signal)
    output = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False

    for i in range(n):
        v = smoothed_signal[i]

        if in_long:
            if v >= exit_thresh:
                output[i] = v
            else:
                in_long = False
        elif in_short:
            if v <= -exit_thresh:
                output[i] = v
            else:
                in_short = False

        if not in_long and not in_short:
            if v > entry_thresh:
                in_long = True
                output[i] = v
            elif v < -entry_thresh:
                in_short = True
                output[i] = v

    return output


if HAS_NUMBA:
    @njit(cache=True)
    def apply_decay_exit(smoothed_signal, entry_thresh, exit_thresh):
        """Numba-accelerated hysteresis decay exit."""
        n = len(smoothed_signal)
        output = np.zeros(n, dtype=np.float64)
        in_long = False
        in_short = False

        for i in range(n):
            v = smoothed_signal[i]

            if in_long:
                if v >= exit_thresh:
                    output[i] = v
                else:
                    in_long = False
            elif in_short:
                if v <= -exit_thresh:
                    output[i] = v
                else:
                    in_short = False

            if not in_long and not in_short:
                if v > entry_thresh:
                    in_long = True
                    output[i] = v
                elif v < -entry_thresh:
                    in_short = True
                    output[i] = v

        return output
else:
    apply_decay_exit = _apply_decay_exit_python


# ── Sweep Parameters ──

SMOOTH_WINDOWS = [10, 50, 100]
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5]
EXIT_THRESHOLDS = [0.0, 0.5, 1.0]
VOL_GATES = [0, 50, 70]

# Rust sim conviction threshold — set LOW since we pre-filter in signal
SIM_CONV_THRESHOLDS = [0.1]


# ── Simulation ──

def run_sim(mbo_file, pred_file, output_file, conv_thresh):
    """Run Rust fill sim with fixed params (chase 1t/3r, 60min hold, no stops)."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(MAX_HOLD_MS),
        '--signal-threshold', str(conv_thresh),
        '--latency-ms', '0',
        '--chase-entry',
        '--chase-max-ticks', '1',
        '--chase-max-reprices', '3',
        '--quiet',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Sim error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Conviction Decay Exit Sweep')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--skip-pred-gen', action='store_true',
                        help='Skip prediction generation (use existing files)')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("CONVICTION DECAY EXIT SWEEP — Dynamic Exit via Signal Zeroing")
    log.info(f"Smooth windows: {SMOOTH_WINDOWS}")
    log.info(f"Entry thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"Exit thresholds: {EXIT_THRESHOLDS}")
    log.info(f"Vol gates: {VOL_GATES}")
    log.info(f"Max hold: {MAX_HOLD_MS}ms (safety net)")
    log.info(f"Workers: {args.workers}")
    log.info(f"Numba available: {HAS_NUMBA}")
    log.info("=" * 80)

    # Warm up numba JIT if available
    if HAS_NUMBA:
        log.info("Warming up numba JIT...")
        _dummy = apply_decay_exit(np.array([0.0, 1.0, 2.0, 3.0, 0.5, -1.0, -2.0, -0.3]), 1.5, 0.5)
        log.info("Numba JIT ready.")

    # Load WF predictions
    log.info("Loading WF predictions...")
    wf_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Count valid combos (exit < entry)
    valid_combos = [(sw, et, xt, vg)
                    for sw in SMOOTH_WINDOWS
                    for et in ENTRY_THRESHOLDS
                    for xt in EXIT_THRESHOLDS
                    for vg in VOL_GATES
                    if xt < et]
    log.info(f"Valid parameter combos: {len(valid_combos)} (exit < entry filter)")
    log.info(f"Prediction files to generate: {len(valid_combos) * len(dates)}")

    # ── Step 1: Generate prediction files with decay exit logic ──
    saved = {}  # (date, label) -> filepath
    gen_t0 = time.time()

    if not args.skip_pred_gen:
        log.info("Generating prediction files with decay exit logic...")

        for di, date in enumerate(dates):
            dt0 = time.time()
            preds_raw = wf_data[f'{date}_preds']
            mid = wf_data[f'{date}_mid']
            n = len(preds_raw)
            if n < 5000:
                continue

            # 1. CNN offset alignment
            aligned = np.zeros(n, dtype=np.float64)
            end = min(n, len(preds_raw) + CNN_OFFSET)
            aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

            # 2. Expanding z-score (vectorized)
            z_scores = zscore_expanding_fast(aligned)

            # 3. Time mask
            tmask = time_mask(n)

            # 4. Vol computation (vectorized)
            vol = compute_trailing_vol(mid)

            # Precompute vol thresholds for needed percentiles
            vol_thresh = {}
            for vg in VOL_GATES:
                if vg > 0:
                    vol_thresh[vg] = compute_expanding_vol_percentile(vol, vg)

            # Precompute vol masks (vectorized boolean arrays)
            vol_masks = {}
            for vg in VOL_GATES:
                if vg == 0:
                    vol_masks[vg] = np.ones(n, dtype=bool)
                else:
                    # Pass vol gate: vol >= threshold AND vol is not NaN
                    valid_vol = ~np.isnan(vol)
                    above_thresh = np.where(valid_vol, vol >= vol_thresh[vg], False)
                    vol_masks[vg] = above_thresh

            # For each smoothing window
            for smooth_w in SMOOTH_WINDOWS:
                smoothed = rolling_mean_smooth(z_scores, smooth_w)
                smoothed = np.nan_to_num(smoothed, nan=0.0)

                for entry_t in ENTRY_THRESHOLDS:
                    for exit_t in EXIT_THRESHOLDS:
                        if exit_t >= entry_t:
                            continue

                        # Apply decay exit logic
                        decay_signal = apply_decay_exit(smoothed, entry_t, exit_t)

                        for vg in VOL_GATES:
                            sig = decay_signal.copy()

                            # Vol gate (vectorized)
                            sig[~vol_masks[vg]] = 0.0

                            # Time mask (vectorized)
                            sig[~tmask] = 0.0

                            label = f'sm{smooth_w}_ent{entry_t}_ext{exit_t}_vol{vg}'
                            fname = f'{date}_{label}.npz'
                            fpath = PRED_OUT_DIR / fname
                            np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
                            saved[(date, label)] = str(fpath)

            dt_elapsed = time.time() - dt0
            if (di + 1) % 5 == 0 or di == 0:
                total_elapsed = time.time() - gen_t0
                rate = (di + 1) / total_elapsed
                eta = (len(dates) - di - 1) / rate if rate > 0 else 0
                log.info(f"  Pred gen: {di+1}/{len(dates)} dates ({dt_elapsed:.1f}s/date), "
                         f"{len(saved)} files, ETA {eta:.0f}s")

        log.info(f"Generated {len(saved)} prediction files in {time.time()-gen_t0:.0f}s")
    else:
        # Load existing prediction files
        log.info("Loading existing prediction files...")
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            try:
                idx = stem.index('_sm')
                date_str = stem[:idx]
                label = stem[idx+1:]
                saved[(date_str, label)] = str(f)
            except ValueError:
                continue
        log.info(f"Found {len(saved)} existing prediction files")

    # ── Step 2: Build sim jobs ──
    log.info("Building simulation jobs...")

    # Cache MBO file lookups
    mbo_cache = {}
    for date in set(d for d, _ in saved.keys()):
        date_compact = date.replace('-', '')
        candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn.zst'))
        if not candidates:
            candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn'))
        if candidates:
            mbo_cache[date] = candidates[0]

    jobs = []
    skipped = 0
    for (date, label), pred_file in saved.items():
        if date not in mbo_cache:
            continue
        mbo = mbo_cache[date]

        for conv in SIM_CONV_THRESHOLDS:
            out_name = f'{label}_conv{int(conv*10)}_{date}.json'
            out_file = SIM_OUT_DIR / out_name
            if out_file.exists():
                skipped += 1
                continue
            jobs.append((str(mbo), pred_file, str(out_file), conv, label, date))

    log.info(f"Total sim jobs: {len(jobs)} (skipped existing: {skipped})")

    if not jobs:
        log.info("No jobs to run. Checking for existing results...")
        # Still aggregate if we have existing result files
        for f in SIM_OUT_DIR.glob('*.json'):
            pass

    # ── Step 3: Run sims ──
    completed = 0
    results = []
    t0 = time.time()

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for mbo, pred, out, conv, label, date in jobs:
                f = executor.submit(run_sim, mbo, pred, out, conv)
                futures[f] = (label, date, conv)

            for future in as_completed(futures):
                label, date, conv = futures[future]
                completed += 1
                res = future.result()
                if res:
                    results.append({
                        'label': label,
                        'date': date,
                        'conv': conv,
                        'pnl': res.get('total_pnl_dollars', 0),
                        'trades': res.get('total_trades', 0),
                        'signals': res.get('total_signals', 0),
                        'filled': res.get('total_filled', 0),
                        'wr': res.get('win_rate', 0),
                        'avg_hold_ms': res.get('avg_hold_ms', 0),
                    })

                if completed % 100 == 0:
                    el = time.time() - t0
                    rate = completed / el if el > 0 else 0
                    eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
                    pnl_so_far = sum(r['pnl'] for r in results)
                    log.info(f"  {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta:.1f}min) "
                             f"| {len(results)} w/trades | cumulative P&L ${pnl_so_far:,.0f}")

        log.info(f"Completed {completed} jobs in {time.time()-t0:.0f}s")

    # Also load any previously completed results from disk
    existing_results = []
    for f in SIM_OUT_DIR.glob('*.json'):
        try:
            stem = f.stem
            # Parse: {label}_conv{X}_{date}.json
            # label is like sm10_ent1.5_ext0.0_vol0
            # Find the _conv part from the right
            parts = stem.rsplit('_', 1)  # date
            if len(parts) != 2:
                continue
            date = parts[1]
            rest = parts[0]
            # Find _conv from the right in rest
            ci = rest.rfind('_conv')
            if ci < 0:
                continue
            label = rest[:ci]
            with open(f) as fh:
                res = json.load(fh)
            existing_results.append({
                'label': label,
                'date': date,
                'conv': 0.1,
                'pnl': res.get('total_pnl_dollars', 0),
                'trades': res.get('total_trades', 0),
                'signals': res.get('total_signals', 0),
                'filled': res.get('total_filled', 0),
                'wr': res.get('win_rate', 0),
                'avg_hold_ms': res.get('avg_hold_ms', 0),
            })
        except Exception:
            continue

    # Merge: prefer fresh results, add existing for anything not in fresh
    fresh_keys = {(r['label'], r['date']) for r in results}
    for er in existing_results:
        if (er['label'], er['date']) not in fresh_keys:
            results.append(er)

    log.info(f"Total results (fresh + existing): {len(results)}")

    # ── Step 4: Aggregate by parameter combo ──
    agg = defaultdict(list)
    for r in results:
        agg[r['label']].append(r)

    summaries = []
    for label, days in agg.items():
        total_pnl = sum(d['pnl'] for d in days)
        total_trades = sum(d['trades'] for d in days)
        total_signals = sum(d['signals'] for d in days)
        total_filled = sum(d['filled'] for d in days)
        n_days = len(days)
        daily_pnls = [d['pnl'] for d in days]
        hold_vals = [d['avg_hold_ms'] for d in days if d.get('avg_hold_ms', 0) > 0]
        avg_hold = np.mean(hold_vals) if hold_vals else 0

        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        std_daily = np.std(daily_pnls) if n_days > 1 else 1
        sharpe = avg_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
        win_rate = sum(d['wr'] * d['trades'] for d in days) / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        pct_profitable_days = sum(1 for p in daily_pnls if p > 0) / max(n_days, 1)

        # Parse label: sm10_ent1.5_ext0.0_vol0
        try:
            parts = label.split('_')
            smooth_w = int(parts[0][2:])
            entry_t = float(parts[1][3:])
            exit_t = float(parts[2][3:])
            vol_gate = int(parts[3][3:])
        except (IndexError, ValueError):
            continue

        summaries.append({
            'label': label,
            'smooth_window': smooth_w,
            'entry_threshold': entry_t,
            'exit_threshold': exit_t,
            'vol_gate': vol_gate,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'pct_profitable_days': round(pct_profitable_days, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'annualized_pnl': round(avg_daily * 252, 0),
            'avg_hold_min': round(avg_hold / 60000, 1) if avg_hold > 0 else 0,
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    # ── Step 5: Print results ──
    log.info(f"\n{'='*120}")
    log.info("CONVICTION DECAY EXIT SWEEP RESULTS — Top 40 by Sharpe")
    log.info(f"{'='*120}")
    log.info(f"{'#':>3} {'Label':<35} {'Sharpe':>7} {'P&L':>12} {'Trades':>6} "
             f"{'WR':>6} {'Fill%':>6} {'ProfDays':>8} {'AvgHold':>8} {'Ann$':>10}")
    log.info("-" * 120)
    for i, s in enumerate(summaries[:40]):
        log.info(f"#{i+1:>2} {s['label']:<35} {s['sharpe']:>7.2f} "
                 f"${s['total_pnl']:>10,.2f} {s['n_trades']:>6d} "
                 f"{s['win_rate']*100:>5.1f}% {s['fill_rate']*100:>5.1f}% "
                 f"{s['pct_profitable_days']*100:>6.1f}% "
                 f"{s['avg_hold_min']:>6.1f}m ${s['annualized_pnl']:>9,.0f}")

    # Parameter sensitivity analysis
    log.info(f"\n{'='*80}")
    log.info("PARAMETER SENSITIVITY (mean Sharpe across other params)")
    log.info(f"{'='*80}")

    for sw in SMOOTH_WINDOWS:
        subset = [s for s in summaries if s['smooth_window'] == sw]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Smooth={sw:>3}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for et in ENTRY_THRESHOLDS:
        subset = [s for s in summaries if s['entry_threshold'] == et]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Entry={et:.1f}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for xt in EXIT_THRESHOLDS:
        subset = [s for s in summaries if s['exit_threshold'] == xt]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Exit={xt:.1f}:  mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for vg in VOL_GATES:
        subset = [s for s in summaries if s['vol_gate'] == vg]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Vol={vg:>2}:    mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    if summaries:
        hold_vals = [s['avg_hold_min'] for s in summaries if s['avg_hold_min'] > 0]
        if hold_vals:
            avg_hold_all = np.mean(hold_vals)
            log.info(f"\n  Average hold time across all configs: {avg_hold_all:.1f} min "
                     f"(vs 30 min fixed hold baseline)")

    # Save results
    out_path = RESULTS_DIR / f'decay_exit_sweep_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'description': 'Conviction Decay Exit Sweep — dynamic exit via signal zeroing',
            'params': {
                'smooth_windows': SMOOTH_WINDOWS,
                'entry_thresholds': ENTRY_THRESHOLDS,
                'exit_thresholds': EXIT_THRESHOLDS,
                'vol_gates': VOL_GATES,
                'max_hold_ms': MAX_HOLD_MS,
                'sim_conv_thresholds': SIM_CONV_THRESHOLDS,
                'chase': '1t/3r',
            },
            'n_combos': len(summaries),
            'summaries': summaries,
        }, f, indent=2)
    log.info(f"\nResults saved to {out_path}")
    log.info("DONE.")


if __name__ == '__main__':
    main()
