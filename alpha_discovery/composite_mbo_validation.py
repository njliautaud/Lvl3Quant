"""
Composite MEAN_ZSCORE -> Rust MBO Fill Sim Validation
=====================================================
1. Generate per-day composite prediction NPZ (mean of 9+ signals)
2. Upload to Jupiter via SFTP
3. Run fill_sim_cli on Jupiter for each day
4. Download results and aggregate

This is the GOLD STANDARD test -- real FIFO queue simulation with MBO data.
"""

import sys
import os
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add teleclaude for SSH utilities
TELECLAUDE = Path(r'C:\Users\Footb\Documents\Github\teleclaude-main')
sys.path.insert(0, str(TELECLAUDE))
from utils.ssh_exec import connect_jupiter, sftp_upload, sftp_download

LVL3_ROOT = Path(r'C:\Users\Footb\Documents\Github\Lvl3Quant')
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
LOCAL_OUT = LVL3_ROOT / 'alpha_discovery' / 'results'

JUPITER_BINARY = '/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli'
JUPITER_MBO_DIR = '/home/jupiter/lvl3quant/data/mbo'
JUPITER_PRED_DIR = '/home/jupiter/lvl3quant/data/predictions_composite'
JUPITER_RESULTS_DIR = '/home/jupiter/lvl3quant/results/composite_mbo'

# Best config from market order sim: t3.5, hold=300s, trail=0, cooldown=50
# Test multiple configs to confirm
CONFIGS = [
    # (name, threshold, hold_ms, trailing_ticks, extra_flags)
    ('best_t3.5_h300s',     3.5, 300000, 0, ''),
    ('t3.0_h300s',          3.0, 300000, 0, ''),
    ('t4.0_h300s',          4.0, 300000, 0, ''),
    ('t3.5_h120s',          3.5, 120000, 0, ''),
    ('t3.5_h600s',          3.5, 600000, 0, ''),
    ('t3.5_h300s_trail4',   3.5, 300000, 4, ''),
    ('t3.5_h300s_prime',    3.5, 300000, 0, '--prime-hours'),
    ('t3.0_h120s',          3.0, 120000, 0, ''),
    ('t2.5_h300s',          2.5, 300000, 0, ''),
    ('t3.5_h300s_lat5',     3.5, 300000, 0, '--latency-ms 5'),
    ('t3.5_h300s_lat20',    3.5, 300000, 0, '--latency-ms 20'),
]


def get_signal_names():
    """Find all signal names with >=80 days available."""
    sig_names_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_names_map[sig_name].add(date)
                break
    return {k: v for k, v in sig_names_map.items() if len(v) >= 80}


def generate_composite_predictions(dates, signal_names):
    """Generate MEAN_ZSCORE composite for each date. Memory-efficient."""
    composites = {}
    sig_list = sorted(signal_names.keys())

    for di, date in enumerate(dates):
        # Get n_bars from first available signal (avoid loading huge feature matrix)
        n_bars = None
        for sig_name in sig_list:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if path.exists():
                try:
                    p = np.load(str(path))['predictions']
                    n_bars = len(p)
                    del p
                    break
                except Exception:
                    continue

        if n_bars is None:
            # Fallback: load feature file
            feat_path = FEAT_CACHE / f'{date}_mbo_features.npz'
            if not feat_path.exists():
                continue
            feats = np.load(str(feat_path), mmap_mode='r')['mbo_features']
            n_bars = len(feats)
            del feats

        sig_sum = np.zeros(n_bars, dtype=np.float64)
        n_sigs = 0

        for sig_name in sig_list:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if not path.exists():
                continue
            try:
                preds = np.load(str(path))['predictions']
            except Exception:
                continue

            if len(preds) != n_bars:
                preds = preds[:n_bars] if len(preds) > n_bars else np.pad(preds, (0, n_bars - len(preds)))

            sig_sum += preds.astype(np.float64)
            n_sigs += 1

        if n_sigs > 0:
            composites[date] = (sig_sum / n_sigs).astype(np.float32)

        if (di + 1) % 20 == 0 or di == len(dates) - 1:
            print(f"    {di + 1}/{len(dates)} days processed ({n_sigs} signals)")

    return composites


def get_available_mbo_dates():
    """Get dates that have MBO files on Jupiter."""
    r = connect_jupiter(f'ls {JUPITER_MBO_DIR}/', timeout=15)
    if not r['success']:
        print(f"ERROR: Can't list Jupiter MBO dir: {r['error']}")
        return []

    dates = []
    for line in r['stdout'].strip().split('\n'):
        line = line.strip()
        if line.startswith('glbx-mdp3-') and line.endswith('.mbo.dbn'):
            # Extract date: glbx-mdp3-20250714.mbo.dbn -> 2025-07-14
            raw = line.replace('glbx-mdp3-', '').replace('.mbo.dbn', '')
            if len(raw) == 8:
                dates.append(f'{raw[:4]}-{raw[4:6]}-{raw[6:8]}')
    return sorted(dates)


def upload_predictions(composites):
    """Upload composite prediction NPZs to Jupiter via single SFTP session."""
    import paramiko

    # Create remote dir
    connect_jupiter(f'mkdir -p {JUPITER_PRED_DIR}', timeout=10)

    # Use a single SFTP session for all uploads (much faster)
    config_path = os.path.join(os.path.dirname(TELECLAUDE), 'teleclaude-main', 'config', 'remote_servers.json')
    # Actually use the connect_jupiter infrastructure
    from utils.ssh_exec import load_config
    config = load_config()
    server = config['servers']['jupiter']

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server['host'], port=22, username=server['username'],
                   password=server['password'], timeout=10,
                   allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    uploaded = 0
    failed = 0
    total = len(composites)

    for date, preds in sorted(composites.items()):
        tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        tmp.close()
        np.savez_compressed(tmp.name, predictions=preds)

        remote_path = f'{JUPITER_PRED_DIR}/composite_mean_{date}.npz'
        try:
            sftp.put(tmp.name, remote_path)
            uploaded += 1
        except Exception as e:
            print(f"  FAILED upload: {date} - {e}")
            failed += 1
        os.unlink(tmp.name)

        if uploaded % 20 == 0 or (uploaded + failed) == total:
            print(f"    {uploaded + failed}/{total} uploaded")

    sftp.close()
    client.close()
    return uploaded, failed


def run_fill_sim(config_name, threshold, hold_ms, trailing_ticks, extra_flags, mbo_dates, composites):
    """Run fill_sim_cli on Jupiter for all available dates."""
    # Create results dir
    connect_jupiter(f'mkdir -p {JUPITER_RESULTS_DIR}/{config_name}', timeout=10)

    valid_dates = [d for d in mbo_dates if d in composites]

    # Build batch command (run all days sequentially on Jupiter)
    cmds = []
    for date in valid_dates:
        date_compact = date.replace('-', '')
        mbo_file = f'{JUPITER_MBO_DIR}/glbx-mdp3-{date_compact}.mbo.dbn'
        pred_file = f'{JUPITER_PRED_DIR}/composite_mean_{date}.npz'
        out_file = f'{JUPITER_RESULTS_DIR}/{config_name}/{date}.json'

        cmd = (
            f'{JUPITER_BINARY} '
            f'--mbo-file {mbo_file} '
            f'--predictions {pred_file} '
            f'--output {out_file} '
            f'--signal-threshold {threshold} '
            f'--hold-ms {hold_ms} '
        )
        if trailing_ticks > 0:
            cmd += f'--trailing-ticks {trailing_ticks} '
        if extra_flags:
            cmd += f'{extra_flags} '
        cmd += '--quiet'

        cmds.append(cmd)

    # Run in batches of 8 (Jupiter has 16 cores)
    batch_size = 8
    total = len(cmds)
    completed = 0

    for i in range(0, total, batch_size):
        batch = cmds[i:i + batch_size]
        # Run batch in parallel using & and wait
        parallel_cmd = ' & '.join(batch) + ' & wait'

        r = connect_jupiter(parallel_cmd, timeout=300)  # 5 min per batch
        if not r['success']:
            print(f"  Batch {i//batch_size + 1} FAILED: {r['error']}")
        else:
            completed += len(batch)
            print(f"  Batch {i//batch_size + 1}: {completed}/{total} done")

    return completed


def collect_results(config_name, mbo_dates, composites):
    """Collect results from Jupiter using cat over SSH (much faster than per-file SFTP)."""
    valid_dates = [d for d in mbo_dates if d in composites]

    # Cat all result files in one SSH command
    cat_files = ' '.join(f'{JUPITER_RESULTS_DIR}/{config_name}/{d}.json' for d in valid_dates)
    # Use a delimiter between files so we can parse them
    cmd = f'for f in {" ".join(f"{JUPITER_RESULTS_DIR}/{config_name}/{d}.json" for d in valid_dates)}; do echo "---FILE_SEP---"; cat "$f" 2>/dev/null; done'
    r = connect_jupiter(cmd, timeout=60)

    all_results = []
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_fills = 0
    total_signals = 0
    h1_pnl = 0.0
    h2_pnl = 0.0
    mid_idx = len(valid_dates) // 2
    day_pnls = []

    if r['success']:
        chunks = r['stdout'].split('---FILE_SEP---')
        date_idx = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or not chunk.startswith('{'):
                continue
            if date_idx >= len(valid_dates):
                break

            try:
                data = json.loads(chunk)
            except Exception:
                date_idx += 1
                continue

            date = valid_dates[date_idx]
            pnl = data.get('total_pnl_dollars', 0)
            trades = data.get('total_trades', 0)
            wins = data.get('winning_trades', 0)
            fills = data.get('fills', trades)
            signals = data.get('signals_generated', 0)

            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            total_fills += fills
            total_signals += signals
            day_pnls.append(pnl)

            if date_idx < mid_idx:
                h1_pnl += pnl
            else:
                h2_pnl += pnl

            all_results.append({
                'date': date,
                'pnl': pnl,
                'trades': trades,
                'wins': wins,
                'fill_rate': data.get('fill_rate', 0),
            })
            date_idx += 1

    # Calculate Sharpe
    if len(day_pnls) > 1:
        arr = np.array(day_pnls)
        sharpe = (arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'config': config_name,
        'total_pnl': total_pnl,
        'h1_pnl': h1_pnl,
        'h2_pnl': h2_pnl,
        'total_trades': total_trades,
        'win_rate': total_wins / max(total_trades, 1) * 100,
        'fill_rate': total_fills / max(total_signals, 1) * 100 if total_signals > 0 else 0,
        'sharpe': sharpe,
        'days': len(all_results),
        'positive_days': sum(1 for p in day_pnls if p > 0),
        'day_pnls': day_pnls,
        'details': all_results,
    }


def main():
    print("=" * 70)
    print("Composite MEAN_ZSCORE -> Rust MBO Fill Sim Validation")
    print("=" * 70)

    # Step 1: Get signal names and available dates
    print("\n[1/5] Loading signal inventory...")
    signal_names = get_signal_names()
    print(f"  Signals with >=80 days: {len(signal_names)}")
    for name in sorted(signal_names):
        print(f"    {name}: {len(signal_names[name])} days")

    # Step 2: Get MBO dates from Jupiter
    print("\n[2/5] Checking Jupiter MBO files...")
    mbo_dates = get_available_mbo_dates()
    print(f"  MBO files on Jupiter: {len(mbo_dates)} days")
    if not mbo_dates:
        print("  ERROR: No MBO files found!")
        return
    print(f"  Date range: {mbo_dates[0]} to {mbo_dates[-1]}")

    # Step 3: Generate composites
    print("\n[3/5] Generating composite predictions...")
    composites = generate_composite_predictions(mbo_dates, signal_names)
    overlap = [d for d in mbo_dates if d in composites]
    print(f"  Composite predictions: {len(composites)} days")
    print(f"  Overlap with MBO: {len(overlap)} days")

    # Step 4: Upload to Jupiter
    print("\n[4/5] Uploading predictions to Jupiter...")
    uploaded, failed = upload_predictions(composites)
    print(f"  Uploaded: {uploaded}, Failed: {failed}")

    # Step 5: Run fill sim for each config
    print("\n[5/5] Running Rust MBO fill sim on Jupiter...")
    all_config_results = []

    for config_name, thresh, hold_ms, trail, extra in CONFIGS:
        print(f"\n  --- Config: {config_name} (t={thresh}, h={hold_ms}ms, trail={trail}) ---")

        completed = run_fill_sim(config_name, thresh, hold_ms, trail, extra, mbo_dates, composites)
        print(f"  Completed: {completed} days")

        results = collect_results(config_name, mbo_dates, composites)
        all_config_results.append(results)

        both_positive = results['h1_pnl'] > 0 and results['h2_pnl'] > 0
        print(f"  PnL: ${results['total_pnl']:+,.0f}  "
              f"H1=${results['h1_pnl']:+,.0f} H2=${results['h2_pnl']:+,.0f}  "
              f"{results['total_trades']} trades  "
              f"{results['win_rate']:.1f}% win  "
              f"Sharpe={results['sharpe']:.2f}  "
              f"{results['positive_days']}/{results['days']} days+  "
              f"{'BOTH+' if both_positive else ''}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Composite MEAN_ZSCORE - Rust MBO Fill Sim")
    print("=" * 70)
    print(f"{'Config':<25} {'PnL':>10} {'H1':>10} {'H2':>10} {'Trades':>7} {'Win%':>6} {'Sharpe':>7} {'Days+':>6}")
    print("-" * 85)

    for r in sorted(all_config_results, key=lambda x: x['total_pnl'], reverse=True):
        both = '*' if r['h1_pnl'] > 0 and r['h2_pnl'] > 0 else ' '
        print(f"{r['config']:<25} ${r['total_pnl']:>+9,.0f} ${r['h1_pnl']:>+9,.0f} ${r['h2_pnl']:>+9,.0f} "
              f"{r['total_trades']:>7} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
              f"{r['positive_days']:>2}/{r['days']:<2}{both}")

    # Save full results
    out_path = LOCAL_OUT / f'composite_mbo_fillsim_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(str(out_path), 'w') as f:
        json.dump({
            'configs': [{k: v for k, v in r.items() if k != 'day_pnls'} for r in all_config_results],
            'signal_names': list(sorted(signal_names.keys())),
            'mbo_dates': mbo_dates,
            'overlap_dates': overlap,
        }, f, indent=2)
    print(f"\nFull results saved: {out_path}")


if __name__ == '__main__':
    main()
