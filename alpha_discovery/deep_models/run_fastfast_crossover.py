#!/usr/bin/env python3
"""
Fast-Fast MA Crossover Sweep
=============================
Tests TWO fast rolling means for entry AND exit, instead of fast/slow combos
where the slow MA was too sluggish to trigger exits.

Concept:
  1. Compute expanding z-score from raw CNN predictions (standard)
  2. Compute FAST MA (rolling mean of z-score: 10, 20 bars)
  3. Compute MEDIUM MA (rolling mean of z-score: 20, 50 bars) — medium > fast
  4. Signal = fast_ma WHEN (fast_ma > medium_ma AND fast_ma > entry_threshold) ELSE 0
     - Entry: fast MA surges above medium MA AND fast_ma > entry_threshold
     - Stay in: while fast_ma > medium_ma
     - Exit: fast_ma drops below medium_ma -> zero signal (fast enough to trigger in minutes)
  5. Vol gates: 0, 50, 70
  6. Entry thresholds: 1.5, 1.75, 2.0
  7. Time mask: skip first 30min, last 15min

Sweep grid: 3 fast/medium combos x 3 entry x 3 vol = 27 combos x dates

Usage:
    python alpha_discovery/deep_models/run_fastfast_crossover.py
    python alpha_discovery/deep_models/run_fastfast_crossover.py --workers 24
"""

import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_fastfast_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_fastfast_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000  # 60 min safety net

# ── Sweep Parameters ──
# Fast/Medium combos: f10/m20, f10/m50, f20/m50 (fast < medium enforced)
FAST_WINDOWS = [10, 20]
MEDIUM_WINDOWS = [20, 50]
VOL_GATES = [0, 50, 70]
ENTRY_THRESHOLDS = [1.5, 1.75, 2.0]

# Rust sim params (fixed)
SIM_SIGNAL_THRESHOLD = 0.1
SIM_CHASE_TICKS = 1
SIM_CHASE_REPRICES = 3

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('fastfast_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'fastfast_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Signal Processing ──

def compute_trailing_vol(mid, window=3000):
    """5-min trailing volatility in bps. Vectorized."""
    n = len(mid)
    ret_1s = np.zeros(n)
    ret_1s[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1e-10) * 10000
    vol = np.full(n, np.nan)
    cs = np.cumsum(ret_1s)
    cs2 = np.cumsum(ret_1s ** 2)
    idx = np.arange(window, n)
    s = cs[idx] - cs[idx - window]
    s2 = cs2[idx] - cs2[idx - window]
    m = s / window
    vol[window:] = np.sqrt(np.maximum(s2 / window - m * m, 0))
    return vol


def compute_expanding_vol_percentile(vol, pct):
    """Expanding percentile for vol gating."""
    s = pd.Series(vol)
    return s.expanding(min_periods=100).quantile(pct / 100.0).values


def zscore_expanding_fast(arr):
    """Expanding z-score — vectorized."""
    n = len(arr)
    result = np.full(n, 0.0, dtype=np.float64)
    valid = ~np.isnan(arr)
    vals = np.where(valid, arr, 0.0)
    cs = np.cumsum(vals)
    cs2 = np.cumsum(vals ** 2)
    cc = np.cumsum(valid.astype(np.float64))

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
    result[~valid] = 0.0
    return result


def rolling_mean(z_scores, window):
    """Rolling mean smoothing."""
    return pd.Series(z_scores).rolling(window, min_periods=1).mean().values


def time_mask(n_bars):
    """Skip first 30 min, last 15 min of session (6.5hr = 390min)."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 375)


# ── Crossover Logic ──

def _apply_crossover_python(fast_ma, medium_ma, entry_thresh):
    """
    Fast-fast crossover signal logic (pure Python fallback).
    Signal = fast_ma WHEN (fast_ma > medium_ma AND fast_ma > entry_threshold) ELSE 0
    Symmetric for shorts.
    """
    n = len(fast_ma)
    output = np.zeros(n, dtype=np.float64)
    for i in range(n):
        fv = fast_ma[i]
        mv = medium_ma[i]
        # Long: fast > medium AND fast > entry_threshold
        if fv > mv and fv > entry_thresh:
            output[i] = fv
        # Short: fast < medium AND fast < -entry_threshold
        elif fv < mv and fv < -entry_thresh:
            output[i] = fv
    return output


if HAS_NUMBA:
    @njit(cache=True)
    def apply_crossover(fast_ma, medium_ma, entry_thresh):
        n = len(fast_ma)
        output = np.zeros(n, dtype=np.float64)
        for i in range(n):
            fv = fast_ma[i]
            mv = medium_ma[i]
            if fv > mv and fv > entry_thresh:
                output[i] = fv
            elif fv < mv and fv < -entry_thresh:
                output[i] = fv
        return output
else:
    apply_crossover = _apply_crossover_python


# ── Prediction Generation ──

def get_valid_combos():
    """Return list of (fast, medium) tuples where fast < medium."""
    combos = []
    for fw in FAST_WINDOWS:
        for mw in MEDIUM_WINDOWS:
            if fw < mw:
                combos.append((fw, mw))
    return combos


def generate_predictions():
    """Generate all fast-fast crossover prediction files locally."""
    log.info("Loading WF predictions...")
    wf_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    fm_combos = get_valid_combos()
    n_combos = len(fm_combos) * len(VOL_GATES) * len(ENTRY_THRESHOLDS)
    log.info(f"Grid: {len(fm_combos)} fast/medium x {len(ENTRY_THRESHOLDS)} entry x "
             f"{len(VOL_GATES)} vol = {n_combos} combos")
    log.info(f"Fast/medium combos: {fm_combos}")
    log.info(f"Total files to generate: {n_combos * len(dates)}")

    # Warm up numba
    if HAS_NUMBA:
        log.info("Warming up numba JIT...")
        _d = apply_crossover(np.array([1.0, 2.0, 0.5]), np.array([0.5, 1.0, 1.5]), 1.0)
        log.info("Numba ready.")

    saved = {}
    gen_t0 = time.time()

    for di, date in enumerate(dates):
        dt0 = time.time()
        preds_raw = wf_data[f'{date}_preds']
        mid = wf_data[f'{date}_mid']
        n = len(preds_raw)
        if n < 5000:
            log.info(f"  Skipping {date}: only {n} bars")
            continue

        # 1. CNN offset alignment
        aligned = np.zeros(n, dtype=np.float64)
        end = min(n, len(preds_raw) + CNN_OFFSET)
        aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

        # 2. Expanding z-score
        z_scores = zscore_expanding_fast(aligned)
        z_scores = np.nan_to_num(z_scores, nan=0.0)

        # 3. Time mask
        tmask = time_mask(n)

        # 4. Vol computation
        vol = compute_trailing_vol(mid)

        # Precompute vol thresholds
        vol_thresh = {}
        for vg in VOL_GATES:
            if vg > 0:
                vol_thresh[vg] = compute_expanding_vol_percentile(vol, vg)

        # Precompute vol masks
        vol_masks = {}
        for vg in VOL_GATES:
            if vg == 0:
                vol_masks[vg] = np.ones(n, dtype=bool)
            else:
                valid_vol = ~np.isnan(vol)
                vol_masks[vg] = np.where(valid_vol, vol >= vol_thresh[vg], False)

        # 5. Precompute all rolling means (fast + medium)
        all_windows = sorted(set(w for combo in fm_combos for w in combo))
        rolling_mas = {}
        for w in all_windows:
            rolling_mas[w] = rolling_mean(z_scores, w)
            rolling_mas[w] = np.nan_to_num(rolling_mas[w], nan=0.0)

        # 6. Generate all combos
        for fw, mw in fm_combos:
            fast = rolling_mas[fw]
            medium = rolling_mas[mw]

            for entry_t in ENTRY_THRESHOLDS:
                # Apply crossover logic
                crossover_signal = apply_crossover(fast, medium, entry_t)

                for vg in VOL_GATES:
                    sig = crossover_signal.copy()

                    # Vol gate
                    sig[~vol_masks[vg]] = 0.0

                    # Time mask
                    sig[~tmask] = 0.0

                    label = f'f{fw}_m{mw}_ent{entry_t}_vol{vg}'
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
    return saved


# ── Local Sim (Neptune) ──

def run_local_sim(mbo_file, pred_file, output_file):
    """Run a single fill_sim_cli invocation locally. Returns (summary, trades) or None."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(MAX_HOLD_MS),
        '--signal-threshold', str(SIM_SIGNAL_THRESHOLD),
        '--latency-ms', '0',
        '--chase-entry',
        '--chase-max-ticks', str(SIM_CHASE_TICKS),
        '--chase-max-reprices', str(SIM_CHASE_REPRICES),
        '--quiet',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Sim error: {e}")
    return None


def extract_hold_stats(sim_result):
    """Extract hold duration stats from per-trade data."""
    trades = sim_result.get('trades', [])
    if not trades:
        return 0.0, 0.0, 0.0, 0

    hold_durations_ms = []
    for t in trades:
        hold_ns = t.get('hold_duration_ns', 0)
        if hold_ns > 0:
            hold_durations_ms.append(hold_ns / 1_000_000.0)

    if not hold_durations_ms:
        return 0.0, 0.0, 0.0, 0

    arr = np.array(hold_durations_ms)
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)), len(arr)


def run_local_sweep(saved, workers=24):
    """Run sweep locally on Neptune."""
    jobs = []
    mbo_cache = {}
    for date in set(d for d, _ in saved.keys()):
        date_compact = date.replace('-', '')
        candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn.zst'))
        if not candidates:
            candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn'))
        if candidates:
            mbo_cache[date] = candidates[0]

    for (date, label), pred_file in saved.items():
        if date not in mbo_cache:
            continue
        mbo = mbo_cache[date]
        out_file = SIM_OUT_DIR / f'{label}_{date}.json'
        if out_file.exists():
            continue
        jobs.append((str(mbo), pred_file, str(out_file), label, date))

    log.info(f"Local sim jobs: {len(jobs)} (workers: {workers})")

    completed = 0
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for mbo, pred, out, label, date in jobs:
            f = executor.submit(run_local_sim, mbo, pred, out)
            futures[f] = (label, date)

        for future in as_completed(futures):
            label, date = futures[future]
            completed += 1
            res = future.result()
            if res:
                avg_hold, med_hold, std_hold, n_hold = extract_hold_stats(res)
                results.append({
                    'label': label,
                    'date': date,
                    'pnl': res.get('total_pnl_dollars', 0),
                    'trades': res.get('total_trades', 0),
                    'signals': res.get('total_signals', 0),
                    'filled': res.get('total_filled', 0),
                    'wr': res.get('win_rate', 0),
                    'avg_hold_ms': avg_hold,
                    'med_hold_ms': med_hold,
                    'std_hold_ms': std_hold,
                })

            if completed % 50 == 0:
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
                pnl_so_far = sum(r['pnl'] for r in results)
                log.info(f"  {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta:.1f}min) "
                         f"| {len(results)} w/trades | P&L ${pnl_so_far:,.0f}")

    log.info(f"Local sweep done: {completed} jobs in {time.time()-t0:.0f}s")
    return results


# ── Load existing results from disk ──

def load_existing_results():
    """Load existing result JSON files from SIM_OUT_DIR."""
    existing = []
    for f in SIM_OUT_DIR.glob('*.json'):
        try:
            stem = f.stem
            # Parse: {label}_{date}.json where date is YYYY-MM-DD at the end
            # Label may contain underscores, so split from the right on _YYYY-MM-DD
            # The date is always 10 chars: YYYY-MM-DD
            if len(stem) < 12:
                continue
            # Try to find YYYY-MM-DD pattern at end
            potential_date = stem[-10:]
            if len(potential_date) == 10 and potential_date[4] == '-' and potential_date[7] == '-':
                date = potential_date
                label = stem[:-11]  # everything before _YYYY-MM-DD
            else:
                continue

            with open(f) as fh:
                res = json.load(fh)

            avg_hold, med_hold, std_hold, n_hold = extract_hold_stats(res)
            existing.append({
                'label': label,
                'date': date,
                'pnl': res.get('total_pnl_dollars', 0),
                'trades': res.get('total_trades', 0),
                'signals': res.get('total_signals', 0),
                'filled': res.get('total_filled', 0),
                'wr': res.get('win_rate', 0),
                'avg_hold_ms': avg_hold,
                'med_hold_ms': med_hold,
                'std_hold_ms': std_hold,
            })
        except Exception:
            continue
    return existing


# ── Aggregation ──

def aggregate_results(results, min_days=20):
    """Aggregate per-date results into per-config summaries."""
    agg = defaultdict(list)
    for r in results:
        agg[r['label']].append(r)

    summaries = []
    for label, days in agg.items():
        n_days = len(days)
        if n_days < min_days:
            continue

        total_pnl = sum(d['pnl'] for d in days)
        total_trades = sum(d['trades'] for d in days)
        total_signals = sum(d['signals'] for d in days)
        total_filled = sum(d['filled'] for d in days)
        daily_pnls = [d['pnl'] for d in days]

        # Hold time stats (weighted by trade count)
        hold_vals = [d['avg_hold_ms'] for d in days if d.get('avg_hold_ms', 0) > 0]
        med_hold_vals = [d['med_hold_ms'] for d in days if d.get('med_hold_ms', 0) > 0]
        avg_hold = np.mean(hold_vals) if hold_vals else 0
        med_hold = np.mean(med_hold_vals) if med_hold_vals else 0

        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        std_daily = np.std(daily_pnls) if n_days > 1 else 1
        sharpe = avg_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
        win_rate = sum(d['wr'] * d['trades'] for d in days) / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        pct_profitable_days = sum(1 for p in daily_pnls if p > 0) / max(n_days, 1)

        # Parse label: f10_m20_ent1.5_vol0
        try:
            parts = label.split('_')
            fast_w = int(parts[0][1:])
            medium_w = int(parts[1][1:])
            entry_t = float(parts[2][3:])
            vol_gate = int(parts[3][3:])
        except (IndexError, ValueError):
            fast_w = medium_w = 0
            entry_t = 0.0
            vol_gate = 0

        summaries.append({
            'label': label,
            'fast_window': fast_w,
            'medium_window': medium_w,
            'entry_threshold': entry_t,
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
            'med_hold_min': round(med_hold / 60000, 1) if med_hold > 0 else 0,
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)
    return summaries


def print_results(summaries):
    """Print ranked results with hold time analysis."""
    log.info(f"\n{'='*140}")
    log.info("FAST-FAST MA CROSSOVER SWEEP RESULTS — Top 20 by Sharpe (20+ days)")
    log.info(f"{'='*140}")
    log.info(f"{'#':>3} {'Label':<28} {'Sharpe':>7} {'P&L':>12} {'Trades':>6} "
             f"{'WR':>6} {'Fill%':>6} {'ProfD%':>7} {'AvgHold':>8} {'MedHold':>8} {'Ann$':>10}")
    log.info("-" * 140)
    for i, s in enumerate(summaries[:20]):
        log.info(f"#{i+1:>2} {s['label']:<28} {s['sharpe']:>7.2f} "
                 f"${s['total_pnl']:>10,.2f} {s['n_trades']:>6d} "
                 f"{s['win_rate']*100:>5.1f}% {s['fill_rate']*100:>5.1f}% "
                 f"{s['pct_profitable_days']*100:>5.1f}% "
                 f"{s['avg_hold_min']:>6.1f}m {s['med_hold_min']:>6.1f}m "
                 f"${s['annualized_pnl']:>9,.0f}")

    # Hold time analysis — the key question
    log.info(f"\n{'='*100}")
    log.info("HOLD TIME ANALYSIS — Is fast-fast crossover actually triggering exits?")
    log.info(f"{'='*100}")
    log.info("  (60min safety net = 60.0m. If avg hold << 60m, crossover exit IS working)")
    log.info("")

    fm_combos = get_valid_combos()
    for fw, mw in fm_combos:
        subset = [s for s in summaries if s['fast_window'] == fw and s['medium_window'] == mw]
        if subset:
            holds = [s['avg_hold_min'] for s in subset if s['avg_hold_min'] > 0]
            med_holds = [s['med_hold_min'] for s in subset if s['med_hold_min'] > 0]
            avg_h = np.mean(holds) if holds else 0
            med_h = np.mean(med_holds) if med_holds else 0
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  f{fw}/m{mw}: avg hold {avg_h:.1f}m, median hold {med_h:.1f}m | "
                     f"best Sharpe {best['sharpe']:.2f} ({best['label']})")

    # Compare to 60min hold baseline
    all_holds = [s['avg_hold_min'] for s in summaries if s['avg_hold_min'] > 0]
    if all_holds:
        log.info(f"\n  Overall: avg hold {np.mean(all_holds):.1f}m, "
                 f"min {np.min(all_holds):.1f}m, max {np.max(all_holds):.1f}m")
        hitting_safety = sum(1 for h in all_holds if h > 55) / len(all_holds) * 100
        log.info(f"  Configs hitting 60m safety net (>55m avg): {hitting_safety:.0f}%")

    # Parameter sensitivity
    log.info(f"\n{'='*80}")
    log.info("PARAMETER SENSITIVITY (mean Sharpe)")
    log.info(f"{'='*80}")

    for fw, mw in fm_combos:
        subset = [s for s in summaries if s['fast_window'] == fw and s['medium_window'] == mw]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  f{fw}/m{mw}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for et in ENTRY_THRESHOLDS:
        subset = [s for s in summaries if s['entry_threshold'] == et]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Entry={et:.2f}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for vg in VOL_GATES:
        subset = [s for s in summaries if s['vol_gate'] == vg]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Vol={vg:>2}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    # Full table for all configs
    log.info(f"\n{'='*140}")
    log.info("ALL CONFIGS (sorted by Sharpe)")
    log.info(f"{'='*140}")
    log.info(f"{'#':>3} {'Label':<28} {'Days':>4} {'Sharpe':>7} {'P&L':>12} {'Trades':>6} "
             f"{'WR':>6} {'Fill%':>6} {'AvgHold':>8} {'MedHold':>8} {'Ann$':>10}")
    log.info("-" * 140)
    for i, s in enumerate(summaries):
        log.info(f"#{i+1:>2} {s['label']:<28} {s['n_days']:>4d} {s['sharpe']:>7.2f} "
                 f"${s['total_pnl']:>10,.2f} {s['n_trades']:>6d} "
                 f"{s['win_rate']*100:>5.1f}% {s['fill_rate']*100:>5.1f}% "
                 f"{s['avg_hold_min']:>6.1f}m {s['med_hold_min']:>6.1f}m "
                 f"${s['annualized_pnl']:>9,.0f}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Fast-Fast MA Crossover Sweep')
    parser.add_argument('--workers', type=int, default=24,
                        help='Workers for local sim (default: 24)')
    parser.add_argument('--gen-only', action='store_true',
                        help='Only generate predictions, do not run sims')
    parser.add_argument('--skip-pred-gen', action='store_true',
                        help='Skip prediction generation (use existing files)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate existing results, no gen or sim')
    args = parser.parse_args()

    fm_combos = get_valid_combos()
    n_combos = len(fm_combos) * len(VOL_GATES) * len(ENTRY_THRESHOLDS)

    log.info("=" * 80)
    log.info("FAST-FAST MA CROSSOVER SWEEP")
    log.info(f"Fast/Medium combos: {fm_combos}")
    log.info(f"Vol gates: {VOL_GATES}")
    log.info(f"Entry thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"Sim: hold {MAX_HOLD_MS}ms, thresh {SIM_SIGNAL_THRESHOLD}, "
             f"chase {SIM_CHASE_TICKS}t/{SIM_CHASE_REPRICES}r")
    log.info(f"Valid combos: {n_combos}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Numba: {HAS_NUMBA}, Pandas: {HAS_PANDAS}")
    log.info("=" * 80)

    if args.aggregate_only:
        log.info("Aggregate-only mode: loading existing results...")
        results = load_existing_results()
        log.info(f"Loaded {len(results)} existing result records")
        summaries = aggregate_results(results, min_days=20)
        print_results(summaries)

        out_path = RESULTS_DIR / f'fastfast_sweep_results_{_ts}.json'
        with open(out_path, 'w') as f:
            json.dump({
                'timestamp': _ts,
                'description': 'Fast-Fast MA Crossover Sweep',
                'params': {
                    'fast_medium_combos': [[fw, mw] for fw, mw in fm_combos],
                    'vol_gates': VOL_GATES,
                    'entry_thresholds': ENTRY_THRESHOLDS,
                    'max_hold_ms': MAX_HOLD_MS,
                    'sim_signal_threshold': SIM_SIGNAL_THRESHOLD,
                    'chase': f'{SIM_CHASE_TICKS}t/{SIM_CHASE_REPRICES}r',
                },
                'n_combos': len(summaries),
                'summaries': summaries,
            }, f, indent=2)
        log.info(f"\nResults saved to {out_path}")
        log.info("DONE.")
        return

    # Step 1: Generate predictions
    if args.skip_pred_gen:
        log.info("Loading existing prediction files...")
        saved = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            try:
                date_str = stem[:10]  # YYYY-MM-DD
                label = stem[11:]     # everything after date_
                saved[(date_str, label)] = str(f)
            except Exception:
                continue
        log.info(f"Found {len(saved)} existing prediction files")
    else:
        saved = generate_predictions()

    if args.gen_only:
        log.info("Prediction generation complete. Exiting (--gen-only).")
        return

    # Step 2: Run locally on Neptune
    log.info("\n--- Running local sweep on Neptune ---")
    results = run_local_sweep(saved, workers=args.workers)

    # Load any existing results too
    existing = load_existing_results()

    # Merge (avoid duplicates)
    fresh_keys = {(r['label'], r['date']) for r in results}
    for er in existing:
        if (er['label'], er['date']) not in fresh_keys:
            results.append(er)

    log.info(f"Total results: {len(results)}")

    # Aggregate and print
    summaries = aggregate_results(results, min_days=20)
    print_results(summaries)

    # Save
    out_path = RESULTS_DIR / f'fastfast_sweep_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'description': 'Fast-Fast MA Crossover Sweep',
            'params': {
                'fast_medium_combos': [[fw, mw] for fw, mw in fm_combos],
                'vol_gates': VOL_GATES,
                'entry_thresholds': ENTRY_THRESHOLDS,
                'max_hold_ms': MAX_HOLD_MS,
                'sim_signal_threshold': SIM_SIGNAL_THRESHOLD,
                'chase': f'{SIM_CHASE_TICKS}t/{SIM_CHASE_REPRICES}r',
            },
            'n_combos': len(summaries),
            'summaries': summaries,
        }, f, indent=2)
    log.info(f"\nResults saved to {out_path}")
    log.info("DONE.")


if __name__ == '__main__':
    main()
