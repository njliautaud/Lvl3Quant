#!/usr/bin/env python3
"""
Stacked Exit Sweep V2 — MA-Based Exits (NO --signal-flip-exit)
================================================================
V1 used --signal-flip-exit which caused 2-second holds and all-negative results.
V2 uses the SAME prediction files (which encode smooth MA exit logic via hysteresis)
but runs WITHOUT --signal-flip-exit. The Rust sim now holds until:
  1. Take-profit (TP ticks)
  2. Trailing stop (SL ticks)
  3. Max hold timeout (60 min safety)

The MA exit logic is already baked INTO the prediction signal:
  - Signal stays active (non-zero) while smoothed z-score > exit threshold
  - Signal goes to zero when it decays below
  - Rust sim enters on signal, holds for hold-ms or until TP/SL

TP sweep: none, 5, 8, 10, 15, 20 ticks
SL sweep: none, 10, 15, 20, 25 ticks (trailing)
Hold: 3600000ms (60 min safety timeout)
Chase: 1t/3r
Signal threshold: 0.1 (low — conviction already baked into signal)

Prediction files: data/processed/cnn_wf_stacked_predictions/ (1,462 files)
Total jobs: 1,462 x 30 TP/SL combos = 43,860

Usage:
    # Deploy to Jupiter + Saturn
    python alpha_discovery/deep_models/run_stacked_exit_sweep_v2.py --deploy

    # Local run (Neptune)
    python alpha_discovery/deep_models/run_stacked_exit_sweep_v2.py --workers 8

    # Aggregate results from servers
    python alpha_discovery/deep_models/run_stacked_exit_sweep_v2.py --aggregate
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

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_stacked_v2_results'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
for d in [SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50

# ── Sweep Grid ──
TP_VALUES = [None, 5, 8, 10, 15, 20]
SL_VALUES = [None, 10, 15, 20, 25]
HOLD_MS = 3600000  # 60 min safety
SIGNAL_THRESHOLD = 0.1

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('stacked_v2')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'stacked_exit_sweep_v2_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ==============================================================================
# LOCAL SIMULATION
# ==============================================================================

def run_single_sim(mbo_file, pred_file, output_file, tp_ticks, sl_ticks):
    """Run one Rust fill_sim job — NO --signal-flip-exit."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(HOLD_MS),
        '--signal-threshold', str(SIGNAL_THRESHOLD),
        '--latency-ms', '0',
        '--chase-entry',
        '--chase-max-ticks', '1',
        '--chase-max-reprices', '3',
        # NO --signal-flip-exit — this is the key fix
        '--quiet',
    ]
    if tp_ticks is not None:
        cmd += ['--take-profit-ticks', str(tp_ticks)]
    if sl_ticks is not None:
        cmd += ['--trailing-ticks', str(sl_ticks)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def build_jobs():
    """Build job list from existing prediction files."""
    pred_files = sorted(PRED_DIR.glob('*.npz'))
    if not pred_files:
        log.error(f"No prediction files found in {PRED_DIR}")
        return []

    log.info(f"Found {len(pred_files)} prediction files")

    jobs = []
    for pf in pred_files:
        stem = pf.stem
        date = stem[:10]
        combo_label = stem[11:]
        nodash = date.replace('-', '')

        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_file.exists():
            continue

        for tp in TP_VALUES:
            for sl in SL_VALUES:
                tp_str = f'tp{tp}' if tp is not None else 'tpN'
                sl_str = f'sl{sl}' if sl is not None else 'slN'
                full_label = f'{combo_label}_{tp_str}_{sl_str}'
                out_file = SIM_OUT_DIR / f'{full_label}_{date}.json'

                if out_file.exists():
                    continue

                jobs.append({
                    'mbo': str(mbo_file),
                    'pred': str(pf),
                    'out': str(out_file),
                    'tp': tp,
                    'sl': sl,
                    'label': full_label,
                    'date': date,
                })

    log.info(f"Total jobs: {len(jobs)} (after skipping existing)")
    return jobs


def run_local_sweep(workers=8):
    """Run sweep locally on Neptune."""
    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        return

    jobs = build_jobs()
    if not jobs:
        log.info("No jobs to run.")
        return

    log.info(f"Running {len(jobs)} jobs with {workers} workers")
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            f = executor.submit(
                run_single_sim,
                job['mbo'], job['pred'], job['out'],
                job['tp'], job['sl']
            )
            futures[f] = job

        for future in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min")

    log.info(f"Done: {done} jobs in {time.time()-t0:.0f}s")


# ==============================================================================
# REMOTE DEPLOYMENT (Jupiter + Saturn)
# ==============================================================================

# The remote sweep script — deployed to both servers
REMOTE_SWEEP_SCRIPT = r'''#!/usr/bin/env python3
"""Stacked Exit Sweep V2 — Remote worker (NO --signal-flip-exit)."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = {workers}
LVL3_ROOT = Path("{lvl3_root}")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_v2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_VALUES = [None, 5, 8, 10, 15, 20]
SL_VALUES = [None, 10, 15, 20, 25]

def run_sim(mbo, pred, out, tp, sl):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", "3600000", "--signal-threshold", "0.1",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", "1", "--chase-max-reprices", "3"]
    # NO --signal-flip-exit — MA exit is encoded in prediction signal
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--trailing-ticks", str(sl)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out).exists():
            return True
    except:
        pass
    return False

pred_files = sorted(PRED_DIR.glob("*.npz"))
print(f"Found {{len(pred_files)}} prediction files", flush=True)

if not BINARY.exists():
    print(f"ERROR: Binary not found: {{BINARY}}", flush=True)
    sys.exit(1)

jobs = []
skipped = 0
for pf in pred_files:
    stem = pf.stem
    date = stem[:10]
    combo_label = stem[11:]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{{nodash}}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{{nodash}}.mbo.dbn"
    if not mbo.exists():
        continue
    for tp in TP_VALUES:
        for sl in SL_VALUES:
            tp_str = f"tp{{tp}}" if tp is not None else "tpN"
            sl_str = f"sl{{sl}}" if sl is not None else "slN"
            full_label = f"{{combo_label}}_{{tp_str}}_{{sl_str}}"
            out_file = OUT_DIR / f"{{full_label}}_{{date}}.json"
            if out_file.exists():
                skipped += 1
                continue
            jobs.append((str(mbo), str(pf), str(out_file), tp, sl))

print(f"Jobs to run: {{len(jobs)}} (skipped {{skipped}} existing)", flush=True)
if not jobs:
    print("No jobs to run — all done!", flush=True)
    sys.exit(0)

done = 0
failed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {{}}
    for mbo, pred, out, tp, sl in jobs:
        f = executor.submit(run_sim, mbo, pred, out, tp, sl)
        futures[f] = out

    for future in as_completed(futures):
        done += 1
        try:
            if not future.result():
                failed += 1
        except:
            failed += 1
        if done % 500 == 0 or done == len(jobs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
            print(f"  [{{done}}/{{len(jobs)}}] {{rate:.1f}}/s, ETA {{eta:.1f}}min, {{failed}} failed", flush=True)

elapsed = time.time() - t0
print(f"DONE: {{done}} jobs in {{elapsed:.0f}}s ({{failed}} failed)", flush=True)
results_count = len(list(OUT_DIR.glob("*.json")))
print(f"Total result files: {{results_count}}", flush=True)
'''


def deploy_to_servers():
    """Upload predictions to Jupiter, rsync to Saturn, launch sweeps on both."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    SATURN_USER = 'saturn'
    SATURN_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")

    JUPITER_ROOT = '/home/jupiter/Lvl3Quant'
    SATURN_ROOT = '/home/saturn/Lvl3Quant'
    JUPITER_PRED_DIR = f'{JUPITER_ROOT}/data/processed/cnn_wf_stacked_predictions'
    SATURN_PRED_DIR = f'{SATURN_ROOT}/data/processed/cnn_wf_stacked_predictions'
    JUPITER_OUT_DIR = f'{JUPITER_ROOT}/data/processed/cnn_wf_stacked_v2_results'
    SATURN_OUT_DIR = f'{SATURN_ROOT}/data/processed/cnn_wf_stacked_v2_results'

    local_pred_files = sorted(PRED_DIR.glob('*.npz'))
    log.info(f"Prediction files to upload: {len(local_pred_files)}")

    # ── Step 1: Connect to Jupiter ──
    log.info("Connecting to Jupiter...")
    ssh_jup = paramiko.SSHClient()
    ssh_jup.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_jup.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=15)

    # Create dirs on Jupiter
    ssh_jup.exec_command(f'mkdir -p {JUPITER_PRED_DIR} {JUPITER_OUT_DIR}')
    time.sleep(1)

    # ── Step 2: Upload predictions to Jupiter via SFTP ──
    log.info("Uploading prediction files to Jupiter...")
    sftp_jup = ssh_jup.open_sftp()

    # Check which files already exist
    try:
        existing_jup = set(sftp_jup.listdir(JUPITER_PRED_DIR))
    except:
        existing_jup = set()

    uploaded = 0
    skipped = 0
    for pf in local_pred_files:
        if pf.name in existing_jup:
            skipped += 1
            continue
        try:
            sftp_jup.put(str(pf), f'{JUPITER_PRED_DIR}/{pf.name}')
            uploaded += 1
            if uploaded % 100 == 0:
                log.info(f"  Uploaded {uploaded} files to Jupiter...")
        except Exception as e:
            log.warning(f"  Upload failed {pf.name}: {e}")

    sftp_jup.close()
    log.info(f"Jupiter: uploaded {uploaded}, skipped {skipped} existing")

    # ── Step 3: Deploy sweep script to Jupiter ──
    jupiter_script = REMOTE_SWEEP_SCRIPT.format(
        workers=14,
        lvl3_root=JUPITER_ROOT,
    )
    script_path_jup = f'{JUPITER_ROOT}/run_stacked_v2_sweep.py'

    sftp_jup = ssh_jup.open_sftp()
    with sftp_jup.open(script_path_jup, 'w') as f:
        f.write(jupiter_script)
    sftp_jup.close()
    log.info(f"Jupiter: deployed sweep script to {script_path_jup}")

    # ── Step 4: Connect to Saturn via Jupiter tunnel ──
    log.info("Connecting to Saturn via Jupiter tunnel...")
    transport = ssh_jup.get_transport()
    channel = transport.open_channel('direct-tcpip', ('saturn', 22), ('127.0.0.1', 0))

    ssh_sat = paramiko.SSHClient()
    ssh_sat.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_sat.connect('saturn', username=SATURN_USER, password=SATURN_PW, sock=channel, timeout=15)

    # Create dirs on Saturn
    ssh_sat.exec_command(f'mkdir -p {SATURN_PRED_DIR} {SATURN_OUT_DIR}')
    time.sleep(1)

    # ── Step 5: Rsync predictions from Jupiter to Saturn ──
    log.info("Rsyncing predictions Jupiter -> Saturn...")
    rsync_cmd = (
        f'rsync -avz --progress {JUPITER_PRED_DIR}/ '
        f'{SATURN_USER}@saturn:{SATURN_PRED_DIR}/'
    )
    # Run rsync from Jupiter (needs sshpass for non-interactive)
    rsync_full = f'sshpass -p "{SATURN_PW}" {rsync_cmd}'
    stdin, stdout, stderr = ssh_jup.exec_command(rsync_full, timeout=600)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        err = stderr.read().decode('utf-8', errors='replace')
        log.warning(f"Rsync exit {exit_status}: {err[:500]}")
        # Fallback: direct SFTP from Jupiter to Saturn
        log.info("Trying direct SFTP Jupiter->Saturn fallback...")
        sftp_sat = ssh_sat.open_sftp()
        try:
            existing_sat = set(sftp_sat.listdir(SATURN_PRED_DIR))
        except:
            existing_sat = set()

        # Read Jupiter file list
        sftp_jup2 = ssh_jup.open_sftp()
        jup_files = sftp_jup2.listdir(JUPITER_PRED_DIR)

        sat_uploaded = 0
        sat_skipped = 0
        for fname in jup_files:
            if fname in existing_sat:
                sat_skipped += 1
                continue
            # Copy via local buffer
            try:
                with sftp_jup2.open(f'{JUPITER_PRED_DIR}/{fname}', 'rb') as src:
                    data = src.read()
                with sftp_sat.open(f'{SATURN_PRED_DIR}/{fname}', 'wb') as dst:
                    dst.write(data)
                sat_uploaded += 1
                if sat_uploaded % 100 == 0:
                    log.info(f"  SFTP'd {sat_uploaded} files to Saturn...")
            except Exception as e:
                log.warning(f"  SFTP failed {fname}: {e}")

        sftp_jup2.close()
        sftp_sat.close()
        log.info(f"Saturn: SFTP'd {sat_uploaded}, skipped {sat_skipped} existing")
    else:
        out = stdout.read().decode('utf-8', errors='replace')
        log.info(f"Rsync complete: {out[-200:]}")

    # ── Step 6: Deploy sweep script to Saturn ──
    saturn_script = REMOTE_SWEEP_SCRIPT.format(
        workers=40,
        lvl3_root=SATURN_ROOT,
    )
    script_path_sat = f'{SATURN_ROOT}/run_stacked_v2_sweep.py'

    sftp_sat = ssh_sat.open_sftp()
    with sftp_sat.open(script_path_sat, 'w') as f:
        f.write(saturn_script)
    sftp_sat.close()
    log.info(f"Saturn: deployed sweep script to {script_path_sat}")

    # ── Step 7: Launch on both servers ──
    # Launch Jupiter
    jup_cmd = f'cd {JUPITER_ROOT} && nohup python3 {script_path_jup} > stacked_v2_sweep.log 2>&1 &'
    ssh_jup.exec_command(jup_cmd)
    log.info(f"Jupiter: launched sweep (14 workers)")
    log.info(f"  Monitor: ssh jupiter 'tail -f {JUPITER_ROOT}/stacked_v2_sweep.log'")

    # Launch Saturn
    sat_cmd = f'cd {SATURN_ROOT} && nohup python3 {script_path_sat} > stacked_v2_sweep.log 2>&1 &'
    ssh_sat.exec_command(sat_cmd)
    log.info(f"Saturn: launched sweep (40 workers)")
    log.info(f"  Monitor: ssh saturn 'tail -f {SATURN_ROOT}/stacked_v2_sweep.log'")

    # Cleanup connections
    ssh_sat.close()
    ssh_jup.close()

    log.info("=" * 60)
    log.info("DEPLOYMENT COMPLETE")
    log.info(f"  Jupiter: 14 workers, {script_path_jup}")
    log.info(f"  Saturn: 40 workers, {script_path_sat}")
    log.info(f"  Total ~43,860 jobs split across both servers")
    log.info(f"  Results dir: cnn_wf_stacked_v2_results/")
    log.info(f"  KEY FIX: NO --signal-flip-exit flag")
    log.info("=" * 60)


# ==============================================================================
# AGGREGATION
# ==============================================================================

def aggregate_results(results_dir=None):
    """Aggregate results from local or downloaded result files."""
    if results_dir is None:
        results_dir = SIM_OUT_DIR

    result_files = sorted(Path(results_dir).glob('*.json'))
    log.info(f"Found {len(result_files)} result files in {results_dir}")

    if not result_files:
        return

    # Group by config label
    config_results = defaultdict(dict)
    for rf in result_files:
        stem = rf.stem
        # Format: {combo_label}_{tp_str}_{sl_str}_{date}
        # Date is last 10 chars
        date = stem[-10:]
        label = stem[:-11]  # everything before _YYYY-MM-DD

        try:
            with open(rf) as f:
                res = json.load(f)
            config_results[label][date] = res
        except Exception:
            pass

    log.info(f"Loaded {len(config_results)} unique configs across {len(result_files)} files")

    # Aggregate
    summaries = []
    for config_label, date_results in config_results.items():
        daily_pnls = []
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        all_trade_pnls = []
        exit_counts = defaultdict(int)
        exit_hold_ms = defaultdict(list)
        exit_pnls = defaultdict(list)

        for date_str, res in sorted(date_results.items()):
            day_pnl = res.get('total_pnl_dollars', 0)
            daily_pnls.append(day_pnl)
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1
                    reason = trade.get('exit_reason', 'Unknown')
                    exit_counts[reason] += 1
                    # hold_duration_ns or hold_time_ms depending on binary version
                    hold_ns = trade.get('hold_duration_ns', 0)
                    hold_ms = hold_ns / 1e6 if hold_ns > 0 else trade.get('hold_time_ms', 0)
                    exit_hold_ms[reason].append(hold_ms)
                    exit_pnls[reason].append(pnl)

        n_days = len(daily_pnls)
        if n_days == 0 or total_trades == 0:
            continue

        total_pnl = sum(daily_pnls)
        avg_daily = np.mean(daily_pnls)
        std_daily = np.std(daily_pnls) if n_days > 1 else 1e-8
        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
        win_rate = total_wins / total_trades
        fill_rate = total_filled / total_signals if total_signals > 0 else 0

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_hold_min = np.mean([ms for ms_list in exit_hold_ms.values() for ms in ms_list]) / 60000 if exit_hold_ms else 0

        # Exit distribution
        exit_dist = {}
        for reason in sorted(exit_counts.keys()):
            c = exit_counts[reason]
            exit_dist[reason] = {
                'count': c,
                'pct': round(c / total_trades * 100, 1),
                'avg_hold_min': round(np.mean(exit_hold_ms[reason]) / 60000, 2) if exit_hold_ms[reason] else 0,
                'avg_pnl': round(np.mean(exit_pnls[reason]), 2) if exit_pnls[reason] else 0,
                'win_rate': round(sum(1 for p in exit_pnls[reason] if p > 0) / len(exit_pnls[reason]) * 100, 1) if exit_pnls[reason] else 0,
            }

        # Parse label for TP/SL
        parts = config_label.split('_')
        tp_label = 'tpN'
        sl_label = 'slN'
        for p in parts:
            if p.startswith('tp'):
                tp_label = p
            if p.startswith('sl'):
                sl_label = p

        summaries.append({
            'config': config_label,
            'tp': tp_label,
            'sl': sl_label,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_pnl / TICK_VALUE, 3),
            'avg_hold_min': round(avg_hold_min, 2),
            'max_dd': round(max_dd, 2),
            'annualized': round(avg_daily * 252, 0),
            'exit_dist': exit_dist,
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    if not summaries:
        log.warning("No configs produced trades!")
        return

    # ── Report ──
    log.info("\n" + "=" * 160)
    log.info("STACKED EXIT SWEEP V2 — TOP 30 BY SHARPE (NO --signal-flip-exit)")
    log.info("MA exit encoded in prediction signal. TP + Trailing Stop + 60min Safety on top.")
    log.info("=" * 160)
    log.info(f"{'#':>3} {'Config':<55} {'Sharpe':>7} {'P&L':>10} {'Trades':>6} {'Fill%':>6} "
             f"{'WR%':>5} {'AvgHold':>8} {'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 160)

    for i, s in enumerate(summaries[:30]):
        log.info(
            f"{i+1:>3} {s['config'][:55]:<55} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_trades']:>6} "
            f"{s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"{s['avg_hold_min']:>7.1f}m ${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # ── Exit Distribution for top 30 ──
    log.info("\n" + "=" * 140)
    log.info("EXIT TYPE DISTRIBUTION — Top 30 configs (V2: no SignalFlip)")
    log.info("=" * 140)
    log.info(f"{'#':>3} {'Config':<50} {'TP%':>6} {'Trail%':>7} {'Timeout%':>8} {'AvgHold(min)':>13}")
    log.info("-" * 140)

    for i, s in enumerate(summaries[:30]):
        ed = s.get('exit_dist', {})
        tp_pct = ed.get('TakeProfit', {}).get('pct', 0)
        tr_pct = ed.get('TrailingStop', {}).get('pct', 0)
        ht_pct = ed.get('HoldTimeout', {}).get('pct', 0)
        short_label = s['config'][:50]
        log.info(
            f"{i+1:>3} {short_label:<50} {tp_pct:>5.1f}% "
            f"{tr_pct:>6.1f}% {ht_pct:>7.1f}% {s['avg_hold_min']:>12.1f}"
        )

    # ── TP x SL Heatmap ──
    log.info("\n" + "=" * 100)
    log.info("TP x SL HEATMAP — Average Sharpe across all prediction configs")
    log.info("=" * 100)

    tp_sl_sharpes = defaultdict(list)
    for s in summaries:
        tp_sl_sharpes[(s['tp'], s['sl'])].append(s['sharpe'])

    tp_labels = sorted(set(s['tp'] for s in summaries),
                       key=lambda x: -1 if x == 'tpN' else int(x.replace('tp', '')))
    sl_labels = sorted(set(s['sl'] for s in summaries),
                       key=lambda x: -1 if x == 'slN' else int(x.replace('sl', '')))

    tp_sl_label = 'TP \\ SL'
    header = f"{tp_sl_label:<8}"
    for sl in sl_labels:
        header += f" {sl:>8}"
    log.info(header)
    log.info("-" * (8 + 9 * len(sl_labels)))

    for tp in tp_labels:
        row = f"{tp:<8}"
        for sl in sl_labels:
            key = (tp, sl)
            if key in tp_sl_sharpes:
                avg_s = np.mean(tp_sl_sharpes[key])
                row += f" {avg_s:>8.2f}"
            else:
                row += f" {'---':>8}"
        log.info(row)

    # ── Reference ──
    log.info("\n" + "=" * 100)
    log.info("REFERENCE — Baselines:")
    log.info("  IS best (vol70/conv2.5/1t/3r/30min): Sharpe 3.28, +$15,479/74d, 130 trades")
    log.info("  OOT static:                          Sharpe 1.58, +$4,082/68d, 102 trades")
    log.info("  V1 stacked (with --signal-flip-exit): ALL NEGATIVE (2s holds)")
    log.info("  V2 KEY FIX: No --signal-flip-exit, MA exit baked into prediction signal")
    log.info("=" * 100)

    # ── Save ──
    out_file = RESULTS_DIR / f'stacked_exit_sweep_v2_results_{_ts}.json'
    clean = []
    for s in summaries:
        cs = {k: v for k, v in s.items() if k != 'exit_dist'}
        ed = s.get('exit_dist', {})
        cs['exit_pct_take_profit'] = ed.get('TakeProfit', {}).get('pct', 0)
        cs['exit_pct_trailing_stop'] = ed.get('TrailingStop', {}).get('pct', 0)
        cs['exit_pct_hold_timeout'] = ed.get('HoldTimeout', {}).get('pct', 0)
        clean.append(cs)

    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'version': 'v2_no_signal_flip_exit',
            'n_configs': len(summaries),
            'n_profitable': sum(1 for s in summaries if s['total_pnl'] > 0),
            'tp_values': [str(t) for t in TP_VALUES],
            'sl_values': [str(t) for t in SL_VALUES],
            'summaries': clean,
        }, f, indent=2)
    log.info(f"\nResults saved: {out_file}")

    return summaries


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stacked Exit Sweep V2 (no --signal-flip-exit)')
    parser.add_argument('--workers', type=int, default=8, help='Local parallel workers')
    parser.add_argument('--deploy', action='store_true', help='Deploy to Jupiter + Saturn')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results')
    parser.add_argument('--results-dir', type=str, help='Custom results directory for aggregation')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("STACKED EXIT SWEEP V2 — NO --signal-flip-exit")
    log.info("  MA exit logic baked into prediction files (hysteresis)")
    log.info("  Sweep: TP (none/5/8/10/15/20) x SL (none/10/15/20/25)")
    log.info("  Hold: 60 min safety | Chase: 1t/3r | Threshold: 0.1")
    log.info("=" * 80)

    if args.deploy:
        deploy_to_servers()
    elif args.aggregate:
        rdir = Path(args.results_dir) if args.results_dir else None
        aggregate_results(rdir)
    else:
        run_local_sweep(args.workers)


if __name__ == '__main__':
    main()
