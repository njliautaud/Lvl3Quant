#!/usr/bin/env python3
"""
Multi-Timeframe Rolling Mean Crossover Sweep
=============================================
Tests FAST/SLOW rolling mean z-score crossover for entry AND exit.

Concept:
  1. Compute expanding z-score (standard)
  2. Compute fast MA of z-score (10, 20, 50 bars)
  3. Compute slow MA of z-score (100, 200, 500, 1000 bars)
  4. Signal = fast_ma WHEN (fast_ma > slow_ma AND fast_ma > entry_threshold) ELSE 0
     - Entry: fast MA crosses above slow MA AND fast_ma > entry_threshold
     - Stay in: while fast_ma > slow_ma
     - Exit: fast_ma drops below slow_ma -> zero signal
  5. Vol gates: 0, 50, 70
  6. Entry thresholds: 1.0, 1.5, 2.0
  7. Time mask: skip first 30min, last 15min

Generates predictions on Neptune, uploads to Jupiter + Saturn, launches
Rust fill_sim sweep on both servers.

Sweep grid: 3 fast x 4 slow x 3 vol x 3 entry = 108 combos x ~41 dates = ~4,428 jobs

Usage:
    python alpha_discovery/deep_models/run_crossover_sweep.py
    python alpha_discovery/deep_models/run_crossover_sweep.py --gen-only
    python alpha_discovery/deep_models/run_crossover_sweep.py --deploy-only
    python alpha_discovery/deep_models/run_crossover_sweep.py --workers 24
"""

import os
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
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False

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
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_crossover_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_crossover_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000  # 60 min safety net

# ── Sweep Parameters ──
FAST_WINDOWS = [10, 20, 50]
SLOW_WINDOWS = [100, 200, 500, 1000]
VOL_GATES = [0, 50, 70]
ENTRY_THRESHOLDS = [1.0, 1.5, 2.0]

# Rust sim params (fixed)
SIM_SIGNAL_THRESHOLD = 0.1  # Low — signal is already gated by crossover logic
SIM_CHASE_TICKS = 1
SIM_CHASE_REPRICES = 3

# ── Server Config ──
JUPITER = {
    'host': 'jupiter',
    'port': 22,
    'user': 'jupiter',
    'password': os.environ.get("CLUSTER_SSH_PASSWORD", ""),
    'pred_dir': '/home/jupiter/Lvl3Quant/data/processed/cnn_wf_crossover_predictions',
    'results_dir': '/home/jupiter/Lvl3Quant/data/processed/cnn_wf_crossover_results',
    'mbo_dir': '/home/jupiter/Lvl3Quant/data/raw/mbo',
    'fill_sim': '/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli',
    'workers': 14,
}

SATURN = {
    'host': 'saturn',
    'port': 22,
    'user': 'saturn',
    'password': os.environ.get("CLUSTER_SSH_PASSWORD", ""),
    'pred_dir': '/home/saturn/Lvl3Quant/data/processed/cnn_wf_crossover_predictions',
    'results_dir': '/home/saturn/Lvl3Quant/data/processed/cnn_wf_crossover_results',
    'mbo_dir': '/home/saturn/Lvl3Quant/data/raw/mbo',
    'fill_sim': '/home/saturn/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli',
    'workers': 40,
}

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('crossover_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'crossover_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
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

def _apply_crossover_python(fast_ma, slow_ma, entry_thresh):
    """
    Crossover signal logic (pure Python).
    Signal = fast_ma WHEN (fast_ma > slow_ma AND fast_ma > entry_threshold) ELSE 0
    Symmetric for shorts: signal = fast_ma WHEN (fast_ma < -slow_ma_abs AND fast_ma < -entry_threshold)
    """
    n = len(fast_ma)
    output = np.zeros(n, dtype=np.float64)

    for i in range(n):
        fv = fast_ma[i]
        sv = slow_ma[i]

        # Long: fast > slow AND fast > entry_threshold
        if fv > sv and fv > entry_thresh:
            output[i] = fv
        # Short: fast < slow AND fast < -entry_threshold
        elif fv < sv and fv < -entry_thresh:
            output[i] = fv

    return output


if HAS_NUMBA:
    @njit(cache=True)
    def apply_crossover(fast_ma, slow_ma, entry_thresh):
        n = len(fast_ma)
        output = np.zeros(n, dtype=np.float64)
        for i in range(n):
            fv = fast_ma[i]
            sv = slow_ma[i]
            if fv > sv and fv > entry_thresh:
                output[i] = fv
            elif fv < sv and fv < -entry_thresh:
                output[i] = fv
        return output
else:
    apply_crossover = _apply_crossover_python


# ── Prediction Generation ──

def generate_predictions():
    """Generate all crossover prediction files locally."""
    log.info("Loading WF predictions...")
    wf_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    n_combos = len(FAST_WINDOWS) * len(SLOW_WINDOWS) * len(VOL_GATES) * len(ENTRY_THRESHOLDS)
    log.info(f"Grid: {len(FAST_WINDOWS)} fast x {len(SLOW_WINDOWS)} slow x "
             f"{len(VOL_GATES)} vol x {len(ENTRY_THRESHOLDS)} entry = {n_combos} combos")
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

        # 5. Precompute all rolling means
        fast_mas = {}
        for fw in FAST_WINDOWS:
            fast_mas[fw] = rolling_mean(z_scores, fw)
            fast_mas[fw] = np.nan_to_num(fast_mas[fw], nan=0.0)

        slow_mas = {}
        for sw in SLOW_WINDOWS:
            slow_mas[sw] = rolling_mean(z_scores, sw)
            slow_mas[sw] = np.nan_to_num(slow_mas[sw], nan=0.0)

        # 6. Generate all combos
        for fw in FAST_WINDOWS:
            for sw in SLOW_WINDOWS:
                # Fast must be faster than slow
                if fw >= sw:
                    continue

                fast = fast_mas[fw]
                slow = slow_mas[sw]

                for entry_t in ENTRY_THRESHOLDS:
                    # Apply crossover logic
                    crossover_signal = apply_crossover(fast, slow, entry_t)

                    for vg in VOL_GATES:
                        sig = crossover_signal.copy()

                        # Vol gate
                        sig[~vol_masks[vg]] = 0.0

                        # Time mask
                        sig[~tmask] = 0.0

                        label = f'f{fw}_s{sw}_ent{entry_t}_vol{vg}'
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


# ── Local Sim (for Neptune) ──

def run_local_sim(mbo_file, pred_file, output_file):
    """Run a single fill_sim_cli invocation locally."""
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Sim error: {e}")
    return None


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
                results.append({
                    'label': label,
                    'date': date,
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
                         f"| {len(results)} w/trades | P&L ${pnl_so_far:,.0f}")

    log.info(f"Local sweep done: {completed} jobs in {time.time()-t0:.0f}s")
    return results


# ── Server Deploy ──

def ssh_connect_jupiter():
    """Connect to Jupiter."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(JUPITER['host'], port=JUPITER['port'],
                   username=JUPITER['user'], password=JUPITER['password'], timeout=15)
    return client


def ssh_connect_saturn(jupiter_client):
    """Connect to Saturn via Jupiter tunnel."""
    transport = jupiter_client.get_transport()
    channel = transport.open_channel('direct-tcpip', (SATURN['host'], 22), (JUPITER['host'], 0))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SATURN['host'], username=SATURN['user'],
                   password=SATURN['password'], sock=channel, timeout=15)
    return client


def ssh_run(client, cmd, timeout=60):
    """Run a command via SSH, return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    code = stdout.channel.recv_exit_status()
    return out, err, code


def upload_predictions(client, server_cfg, pred_dir):
    """Upload all prediction NPZ files to a server via SFTP."""
    sftp = client.open_sftp()

    # Ensure remote dir exists
    try:
        sftp.stat(server_cfg['pred_dir'])
    except IOError:
        # mkdir -p via SSH
        ssh_run(client, f"mkdir -p {server_cfg['pred_dir']}")

    pred_files = sorted(Path(pred_dir).glob('*.npz'))
    log.info(f"  Uploading {len(pred_files)} prediction files to {server_cfg['host']}:{server_cfg['pred_dir']}...")

    uploaded = 0
    errors = 0
    t0 = time.time()

    for pf in pred_files:
        remote_path = f"{server_cfg['pred_dir']}/{pf.name}"
        try:
            sftp.put(str(pf), remote_path)
            uploaded += 1
            if uploaded % 200 == 0:
                elapsed = time.time() - t0
                rate = uploaded / elapsed if elapsed > 0 else 0
                eta = (len(pred_files) - uploaded) / rate / 60 if rate > 0 else 0
                log.info(f"    {uploaded}/{len(pred_files)} ({rate:.0f}/s, ETA {eta:.1f}min)")
        except Exception as e:
            errors += 1
            if errors <= 5:
                log.error(f"    Upload error: {pf.name}: {e}")

    sftp.close()
    elapsed = time.time() - t0
    log.info(f"  Upload complete: {uploaded}/{len(pred_files)} in {elapsed:.0f}s ({errors} errors)")
    return uploaded


def generate_sweep_script(server_cfg):
    """Generate the bash sweep script that runs on the server."""
    user = server_cfg['user']
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# Crossover sweep -- auto-generated ' + datetime.now().isoformat())
    lines.append('# Run: nohup bash run_crossover_sweep.sh > crossover_sweep.log 2>&1 &')
    lines.append('')
    lines.append('FILL_SIM="' + server_cfg['fill_sim'] + '"')
    lines.append('PRED_DIR="' + server_cfg['pred_dir'] + '"')
    lines.append('RESULTS_DIR="' + server_cfg['results_dir'] + '"')
    lines.append('MBO_DIR="' + server_cfg['mbo_dir'] + '"')
    lines.append('MAX_WORKERS=' + str(server_cfg['workers']))
    lines.append('HOLD_MS=' + str(MAX_HOLD_MS))
    lines.append('SIGNAL_THRESH=' + str(SIM_SIGNAL_THRESHOLD))
    lines.append('CHASE_TICKS=' + str(SIM_CHASE_TICKS))
    lines.append('CHASE_REPRICES=' + str(SIM_CHASE_REPRICES))
    lines.append('')
    lines.append('mkdir -p "$RESULTS_DIR"')
    lines.append('')
    lines.append('echo "=== Crossover Sweep ==="')
    lines.append('echo "Fill sim: $FILL_SIM"')
    lines.append('echo "Predictions: $PRED_DIR"')
    lines.append('echo "Results: $RESULTS_DIR"')
    lines.append('echo "MBO: $MBO_DIR"')
    lines.append('echo "Workers: $MAX_WORKERS"')
    lines.append('echo "Started: $(date)"')
    lines.append('')
    lines.append('# Check fill_sim exists')
    lines.append('if [ ! -f "$FILL_SIM" ]; then')
    lines.append('    echo "ERROR: fill_sim_cli not found at $FILL_SIM"')
    lines.append('    for alt in \\')
    lines.append('        "/home/' + user + '/lvl3quant/rust_cache_builder/target/release/fill_sim_cli" \\')
    lines.append('        "/home/' + user + '/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"; do')
    lines.append('        if [ -f "$alt" ]; then')
    lines.append('            echo "Found at: $alt"')
    lines.append('            FILL_SIM="$alt"')
    lines.append('            break')
    lines.append('        fi')
    lines.append('    done')
    lines.append('    if [ ! -f "$FILL_SIM" ]; then')
    lines.append('        echo "FATAL: Cannot find fill_sim_cli anywhere. Exiting."')
    lines.append('        exit 1')
    lines.append('    fi')
    lines.append('fi')
    lines.append('')
    lines.append('echo "MBO files available:"')
    lines.append('ls "$MBO_DIR"/*.dbn.zst 2>/dev/null | wc -l')
    lines.append('echo "Prediction files:"')
    lines.append('ls "$PRED_DIR"/*.npz 2>/dev/null | wc -l')
    lines.append('')
    lines.append('# Build job list')
    lines.append('JOBS_FILE=$(mktemp)')
    lines.append('TOTAL=0')
    lines.append('SKIPPED=0')
    lines.append('')
    lines.append('for pred_file in "$PRED_DIR"/*.npz; do')
    lines.append('    [ -f "$pred_file" ] || continue')
    lines.append('    fname=$(basename "$pred_file" .npz)')
    lines.append('')
    lines.append('    # Extract date (YYYY-MM-DD) from start of filename')
    lines.append('    date_str=$(echo "$fname" | grep -oP \'^\\d{4}-\\d{2}-\\d{2}\')')
    lines.append('    [ -z "$date_str" ] && continue')
    lines.append('')
    lines.append('    # Label is everything after date_')
    lines.append('    label=${fname#${date_str}_}')
    lines.append('')
    lines.append('    # Find MBO file')
    lines.append('    date_compact=$(echo "$date_str" | tr -d \'-\')')
    lines.append('    mbo_file=$(ls "$MBO_DIR"/*"$date_compact"*.dbn.zst 2>/dev/null | head -1)')
    lines.append('    [ -z "$mbo_file" ] && mbo_file=$(ls "$MBO_DIR"/*"$date_compact"*.dbn 2>/dev/null | head -1)')
    lines.append('    [ -z "$mbo_file" ] && continue')
    lines.append('')
    lines.append('    out_file="$RESULTS_DIR/${label}_${date_str}.json"')
    lines.append('')
    lines.append('    if [ -f "$out_file" ]; then')
    lines.append('        SKIPPED=$((SKIPPED + 1))')
    lines.append('        continue')
    lines.append('    fi')
    lines.append('')
    lines.append('    echo "$mbo_file|$pred_file|$out_file" >> "$JOBS_FILE"')
    lines.append('    TOTAL=$((TOTAL + 1))')
    lines.append('done')
    lines.append('')
    lines.append('echo "Total jobs: $TOTAL (skipped existing: $SKIPPED)"')
    lines.append('')
    lines.append('if [ "$TOTAL" -eq 0 ]; then')
    lines.append('    echo "Nothing to do. Exiting."')
    lines.append('    rm -f "$JOBS_FILE"')
    lines.append('    exit 0')
    lines.append('fi')
    lines.append('')
    lines.append('# Run jobs in parallel using xargs')
    lines.append('run_one() {')
    lines.append('    local line="$1"')
    lines.append('    local mbo_file=$(echo "$line" | cut -d\'|\' -f1)')
    lines.append('    local pred_file=$(echo "$line" | cut -d\'|\' -f2)')
    lines.append('    local out_file=$(echo "$line" | cut -d\'|\' -f3)')
    lines.append('')
    lines.append('    "$FILL_SIM" \\')
    lines.append('        --mbo-file "$mbo_file" \\')
    lines.append('        --predictions "$pred_file" \\')
    lines.append('        --output "$out_file" \\')
    lines.append('        --hold-ms "$HOLD_MS" \\')
    lines.append('        --signal-threshold "$SIGNAL_THRESH" \\')
    lines.append('        --latency-ms 0 \\')
    lines.append('        --chase-entry \\')
    lines.append('        --chase-max-ticks "$CHASE_TICKS" \\')
    lines.append('        --chase-max-reprices "$CHASE_REPRICES" \\')
    lines.append('        --quiet 2>/dev/null')
    lines.append('')
    lines.append('    if [ $? -ne 0 ]; then')
    lines.append('        echo "FAIL: $(basename "$pred_file")" >&2')
    lines.append('    fi')
    lines.append('}')
    lines.append('export -f run_one')
    lines.append('export FILL_SIM HOLD_MS SIGNAL_THRESH CHASE_TICKS CHASE_REPRICES')
    lines.append('')
    lines.append('echo "Launching $MAX_WORKERS parallel workers..."')
    lines.append('echo "Start time: $(date)"')
    lines.append('')
    lines.append('cat "$JOBS_FILE" | xargs -P "$MAX_WORKERS" -I {} bash -c \'run_one "{}"\'')
    lines.append('')
    lines.append('echo ""')
    lines.append('echo "=== Sweep Complete ==="')
    lines.append('echo "End time: $(date)"')
    lines.append('echo "Results in: $RESULTS_DIR"')
    lines.append('RESULT_COUNT=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)')
    lines.append('echo "Total result files: $RESULT_COUNT"')
    lines.append('')
    lines.append('rm -f "$JOBS_FILE"')
    lines.append('')
    lines.append('echo ""')
    lines.append('echo "=== Quick Stats ==="')
    lines.append('TOTAL_TRADES=0')
    lines.append('for f in "$RESULTS_DIR"/*.json; do')
    lines.append('    [ -f "$f" ] || continue')
    lines.append('    trades=$(python3 -c "import json; d=json.load(open(\'$f\')); print(d.get(\'total_trades\',0))" 2>/dev/null)')
    lines.append('    [ -n "$trades" ] && TOTAL_TRADES=$((TOTAL_TRADES + trades))')
    lines.append('done')
    lines.append('echo "Total trades across all results: $TOTAL_TRADES"')
    lines.append('echo "DONE."')

    return '\n'.join(lines) + '\n'


def deploy_to_server(client, server_cfg, server_name):
    """Deploy sweep script to a server and launch it."""
    log.info(f"  Deploying sweep script to {server_name}...")

    # Generate script
    script_content = generate_sweep_script(server_cfg)

    # Ensure results dir exists
    ssh_run(client, f"mkdir -p {server_cfg['results_dir']}")

    # Upload script via SFTP
    sftp = client.open_sftp()
    script_path = f"{server_cfg['pred_dir']}/../run_crossover_sweep.sh"
    # Normalize path
    out, _, _ = ssh_run(client, f"dirname {server_cfg['pred_dir']}")
    parent_dir = out.strip()
    script_path = f"{parent_dir}/run_crossover_sweep.sh"

    with sftp.open(script_path, 'w') as f:
        f.write(script_content)
    sftp.close()

    ssh_run(client, f"chmod +x {script_path}")
    log.info(f"  Script deployed to {script_path}")

    # Launch in background
    log.info(f"  Launching sweep on {server_name} ({server_cfg['workers']} workers)...")
    log_path = f"{parent_dir}/crossover_sweep.log"
    cmd = f"nohup bash {script_path} > {log_path} 2>&1 &"
    ssh_run(client, cmd)

    # Verify it started
    time.sleep(2)
    out, _, _ = ssh_run(client, f"ps aux | grep run_crossover_sweep | grep -v grep | wc -l")
    procs = out.strip()
    log.info(f"  Processes running: {procs}")

    # Show initial log
    out, _, _ = ssh_run(client, f"head -20 {log_path} 2>/dev/null")
    if out.strip():
        for line in out.strip().split('\n')[:10]:
            log.info(f"    [{server_name}] {line}")

    return script_path, log_path


def deploy_all():
    """Upload predictions to Jupiter and Saturn, deploy and launch sweeps."""
    if not HAS_PARAMIKO:
        log.error("paramiko not installed. Cannot deploy to servers.")
        return False

    # Count local prediction files
    pred_files = list(PRED_OUT_DIR.glob('*.npz'))
    if not pred_files:
        log.error("No prediction files found. Run --gen-only first.")
        return False
    log.info(f"Prediction files to upload: {len(pred_files)}")

    # Connect to Jupiter
    log.info("Connecting to Jupiter...")
    jupiter_client = ssh_connect_jupiter()
    log.info("Connected to Jupiter.")

    # Upload to Jupiter
    upload_predictions(jupiter_client, JUPITER, PRED_OUT_DIR)

    # Deploy to Jupiter
    j_script, j_log = deploy_to_server(jupiter_client, JUPITER, 'Jupiter')

    # Connect to Saturn via Jupiter tunnel
    log.info("Connecting to Saturn via Jupiter tunnel...")
    saturn_client = ssh_connect_saturn(jupiter_client)
    log.info("Connected to Saturn.")

    # Upload to Saturn
    upload_predictions(saturn_client, SATURN, PRED_OUT_DIR)

    # Deploy to Saturn
    s_script, s_log = deploy_to_server(saturn_client, SATURN, 'Saturn')

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("DEPLOYMENT COMPLETE")
    log.info("=" * 70)
    log.info(f"Jupiter: {JUPITER['workers']} workers, log: {j_log}")
    log.info(f"Saturn:  {SATURN['workers']} workers, log: {s_log}")
    log.info("")
    log.info("Monitor progress:")
    log.info(f"  Jupiter: ssh jupiter@{JUPITER['host']} 'tail -f {j_log}'")
    log.info(f"  Saturn:  (via Jupiter) ssh saturn@{SATURN['host']} 'tail -f {s_log}'")
    log.info(f"  Results: ls {JUPITER['results_dir']}/*.json | wc -l")

    # Cleanup
    saturn_client.close()
    jupiter_client.close()

    return True


# ── Aggregation ──

def aggregate_results(results):
    """Aggregate per-date results into per-config summaries."""
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

        # Parse label: f10_s100_ent1.0_vol0
        try:
            parts = label.split('_')
            fast_w = int(parts[0][1:])
            slow_w = int(parts[1][1:])
            entry_t = float(parts[2][3:])
            vol_gate = int(parts[3][3:])
        except (IndexError, ValueError):
            fast_w = slow_w = 0
            entry_t = 0.0
            vol_gate = 0

        summaries.append({
            'label': label,
            'fast_window': fast_w,
            'slow_window': slow_w,
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
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)
    return summaries


def print_results(summaries):
    """Print ranked results and parameter sensitivity."""
    log.info(f"\n{'='*130}")
    log.info("CROSSOVER SWEEP RESULTS — Top 40 by Sharpe")
    log.info(f"{'='*130}")
    log.info(f"{'#':>3} {'Label':<30} {'Sharpe':>7} {'P&L':>12} {'Trades':>6} "
             f"{'WR':>6} {'Fill%':>6} {'ProfDays':>8} {'AvgHold':>8} {'Ann$':>10}")
    log.info("-" * 130)
    for i, s in enumerate(summaries[:40]):
        log.info(f"#{i+1:>2} {s['label']:<30} {s['sharpe']:>7.2f} "
                 f"${s['total_pnl']:>10,.2f} {s['n_trades']:>6d} "
                 f"{s['win_rate']*100:>5.1f}% {s['fill_rate']*100:>5.1f}% "
                 f"{s['pct_profitable_days']*100:>6.1f}% "
                 f"{s['avg_hold_min']:>6.1f}m ${s['annualized_pnl']:>9,.0f}")

    # Parameter sensitivity
    log.info(f"\n{'='*80}")
    log.info("PARAMETER SENSITIVITY (mean Sharpe across other params)")
    log.info(f"{'='*80}")

    for fw in FAST_WINDOWS:
        subset = [s for s in summaries if s['fast_window'] == fw]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Fast={fw:>4}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for sw in SLOW_WINDOWS:
        subset = [s for s in summaries if s['slow_window'] == sw]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Slow={sw:>4}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for et in ENTRY_THRESHOLDS:
        subset = [s for s in summaries if s['entry_threshold'] == et]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Entry={et:.1f}: mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")

    for vg in VOL_GATES:
        subset = [s for s in summaries if s['vol_gate'] == vg]
        if subset:
            avg_sr = np.mean([s['sharpe'] for s in subset])
            best = max(subset, key=lambda x: x['sharpe'])
            log.info(f"  Vol={vg:>2}:   mean Sharpe {avg_sr:.3f}, "
                     f"best {best['sharpe']:.3f} ({best['label']})")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Multi-Timeframe Rolling Mean Crossover Sweep')
    parser.add_argument('--workers', type=int, default=24,
                        help='Workers for local sim (default: 24)')
    parser.add_argument('--gen-only', action='store_true',
                        help='Only generate predictions, do not run sims or deploy')
    parser.add_argument('--deploy-only', action='store_true',
                        help='Only deploy to servers (predictions must exist)')
    parser.add_argument('--local-only', action='store_true',
                        help='Only run locally on Neptune (no server deploy)')
    parser.add_argument('--skip-pred-gen', action='store_true',
                        help='Skip prediction generation (use existing files)')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("MULTI-TIMEFRAME ROLLING MEAN CROSSOVER SWEEP")
    log.info(f"Fast windows: {FAST_WINDOWS}")
    log.info(f"Slow windows: {SLOW_WINDOWS}")
    log.info(f"Vol gates: {VOL_GATES}")
    log.info(f"Entry thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"Sim: hold {MAX_HOLD_MS}ms, thresh {SIM_SIGNAL_THRESHOLD}, "
             f"chase {SIM_CHASE_TICKS}t/{SIM_CHASE_REPRICES}r")
    n_combos = len(FAST_WINDOWS) * len(SLOW_WINDOWS) * len(VOL_GATES) * len(ENTRY_THRESHOLDS)
    # Subtract invalid combos where fast >= slow
    invalid = sum(1 for fw in FAST_WINDOWS for sw in SLOW_WINDOWS if fw >= sw)
    valid_cross = len(FAST_WINDOWS) * len(SLOW_WINDOWS) - invalid
    n_combos = valid_cross * len(VOL_GATES) * len(ENTRY_THRESHOLDS)
    log.info(f"Valid combos: {n_combos} (fast < slow enforced)")
    log.info(f"Workers: {args.workers}")
    log.info(f"Numba: {HAS_NUMBA}, Paramiko: {HAS_PARAMIKO}")
    log.info("=" * 80)

    # Step 1: Generate predictions
    if not args.deploy_only:
        if args.skip_pred_gen:
            log.info("Loading existing prediction files...")
            saved = {}
            for f in PRED_OUT_DIR.glob('*.npz'):
                stem = f.stem
                # Parse: {date}_{label}
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

    # Step 2: Deploy to servers
    if not args.local_only:
        log.info("\n--- Deploying to Jupiter and Saturn ---")
        deploy_all()

    # Step 3: Run locally (optional)
    if args.local_only and not args.deploy_only:
        log.info("\n--- Running local sweep on Neptune ---")
        results = run_local_sweep(saved, workers=args.workers)

        # Load any existing results too
        existing = []
        for f in SIM_OUT_DIR.glob('*.json'):
            try:
                stem = f.stem
                # Parse: {label}_{date}.json
                parts = stem.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                date = parts[1]
                label = parts[0]
                with open(f) as fh:
                    res = json.load(fh)
                existing.append({
                    'label': label, 'date': date,
                    'pnl': res.get('total_pnl_dollars', 0),
                    'trades': res.get('total_trades', 0),
                    'signals': res.get('total_signals', 0),
                    'filled': res.get('total_filled', 0),
                    'wr': res.get('win_rate', 0),
                    'avg_hold_ms': res.get('avg_hold_ms', 0),
                })
            except Exception:
                continue

        # Merge
        fresh_keys = {(r['label'], r['date']) for r in results}
        for er in existing:
            if (er['label'], er['date']) not in fresh_keys:
                results.append(er)

        log.info(f"Total results: {len(results)}")

        # Aggregate and print
        summaries = aggregate_results(results)
        print_results(summaries)

        # Save
        out_path = RESULTS_DIR / f'crossover_sweep_results_{_ts}.json'
        with open(out_path, 'w') as f:
            json.dump({
                'timestamp': _ts,
                'description': 'Multi-Timeframe Rolling Mean Crossover Sweep',
                'params': {
                    'fast_windows': FAST_WINDOWS,
                    'slow_windows': SLOW_WINDOWS,
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
