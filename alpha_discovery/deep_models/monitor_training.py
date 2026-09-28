"""
Auto-monitor deep learning training logs and report via Discord webhook.

Usage:
    python -m alpha_discovery.deep_models.monitor_training --interval 120

Watches for walkforward_*.log files in the results directory and sends
periodic summaries (fold ICs, loss trends, GPU status) to Discord.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Discord webhook URL (from environment or hardcoded for internal use)
DISCORD_WEBHOOK = os.environ.get('DISCORD_WEBHOOK_URL', '')

RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def parse_log_file(log_path: str) -> dict:
    """Parse a walkforward training log for key metrics."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    model_type = 'unknown'
    folds = []
    current_fold = None
    total_folds = 0
    config = {}

    for line in lines:
        line = line.strip()

        # Model type
        m = re.search(r'Walk-Forward Training: (\w+)', line)
        if m:
            model_type = m.group(1)

        # Total folds
        m = re.search(r'Total folds: (\d+)', line)
        if m:
            total_folds = int(m.group(1))

        # Config
        m = re.search(r'epochs/fold:\s+(\d+)', line)
        if m:
            config['epochs'] = int(m.group(1))
        m = re.search(r'batch_size:\s+(\d+)', line)
        if m:
            config['batch_size'] = int(m.group(1))

        # Fold start
        m = re.search(r'Fold (\d+)/(\d+) \| Train:.*\| Test: (\S+)', line)
        if m:
            current_fold = {
                'fold_num': int(m.group(1)),
                'test_date': m.group(3),
                'epochs': [],
            }

        # Target normalization
        m = re.search(r'Target normalization: mean=([\d.-]+) std=([\d.-]+)', line)
        if m and current_fold:
            current_fold['tgt_mean'] = float(m.group(1))
            current_fold['tgt_std'] = float(m.group(2))

        # Epoch result
        m = re.search(r'Epoch (\d+)/(\d+): loss=([\d.]+)\s+IC=([+-]?[\d.]+)', line)
        if m and current_fold:
            current_fold['epochs'].append({
                'epoch': int(m.group(1)),
                'loss': float(m.group(3)),
                'ic': float(m.group(4)),
            })

        # Fold IC
        m = re.search(r'Fold IC: ([+-]?[\d.]+)', line)
        if m and current_fold:
            current_fold['fold_ic'] = float(m.group(1))
            folds.append(current_fold)
            current_fold = None

        # Final results
        m = re.search(r'Per-fold IC mean:\s+([+-]?[\d.]+)', line)
        if m:
            config['final_ic'] = float(m.group(1))

    # If a fold is in progress (no fold IC yet), include it
    in_progress_fold = None
    if current_fold and current_fold.get('epochs'):
        in_progress_fold = current_fold

    return {
        'model_type': model_type,
        'total_folds': total_folds,
        'completed_folds': len(folds),
        'folds': folds,
        'in_progress_fold': in_progress_fold,
        'config': config,
        'log_path': log_path,
    }


def get_gpu_status() -> str:
    """Get GPU utilization info."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader'],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        return out
    except Exception:
        return 'N/A'


def format_report(parsed: dict, gpu_status: str) -> str:
    """Format a Discord-friendly report."""
    p = parsed
    lines = []
    lines.append(f"**Training Monitor: {p['model_type']}**")
    lines.append(f"Folds: {p['completed_folds']}/{p['total_folds']} | GPU: {gpu_status}")

    if p['completed_folds'] > 0:
        ics = [f['fold_ic'] for f in p['folds'] if 'fold_ic' in f]
        avg_ic = sum(ics) / len(ics) if ics else 0
        last3 = ics[-3:] if len(ics) >= 3 else ics
        lines.append(f"Avg IC: {avg_ic:+.4f} | Last 3: {', '.join(f'{x:+.4f}' for x in last3)}")

        # Loss trend from last completed fold
        last_fold = p['folds'][-1]
        if last_fold.get('epochs'):
            losses = [e['loss'] for e in last_fold['epochs']]
            ics_epoch = [e['ic'] for e in last_fold['epochs']]
            lines.append(f"Last fold losses: {' → '.join(f'{l:.4f}' for l in losses)}")
            lines.append(f"Last fold ICs:    {' → '.join(f'{i:+.4f}' for i in ics_epoch)}")
            if last_fold.get('tgt_std'):
                lines.append(f"Target std: {last_fold['tgt_std']:.3f} ticks (loss is on z-scored targets)")

    # In-progress fold
    if p['in_progress_fold']:
        ipf = p['in_progress_fold']
        fold_num = ipf.get('fold_num', '?')
        n_epochs = len(ipf['epochs'])
        lines.append(f"Currently training fold {fold_num}, epoch {n_epochs}...")
        if ipf['epochs']:
            last_ep = ipf['epochs'][-1]
            lines.append(f"  Latest: loss={last_ep['loss']:.4f} IC={last_ep['ic']:+.4f}")

    # Check for red flags
    if p['completed_folds'] >= 3:
        ics = [f['fold_ic'] for f in p['folds'] if 'fold_ic' in f]
        if all(ic < 0.05 for ic in ics[-3:]):
            lines.append("⚠️ **RED FLAG**: Last 3 fold ICs all below 0.05 — model may not be learning")
        avg_loss = 0
        for fold in p['folds'][-3:]:
            if fold.get('epochs'):
                ep_losses = [e['loss'] for e in fold['epochs']]
                if len(ep_losses) >= 2 and ep_losses[-1] > ep_losses[0]:
                    lines.append(f"⚠️ Fold {fold['fold_num']}: loss INCREASING ({ep_losses[0]:.4f} → {ep_losses[-1]:.4f})")

    return '\n'.join(lines)


def send_discord(message: str):
    """Send message to Discord via webhook."""
    if not DISCORD_WEBHOOK:
        print(f"[monitor] No webhook set. Message:\n{message}")
        return
    import urllib.request
    data = json.dumps({'content': message}).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[monitor] Discord send failed: {e}")


def find_latest_log() -> str | None:
    """Find the most recently modified walkforward log."""
    pattern = str(RESULTS_DIR / 'walkforward_*.log')
    logs = glob.glob(pattern)
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='Monitor DL training and report to Discord')
    parser.add_argument('--interval', type=int, default=120,
                        help='Check interval in seconds (default: 120)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Specific log file to monitor (default: latest)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (no loop)')
    args = parser.parse_args()

    print(f"[monitor] Starting training monitor (interval={args.interval}s)")
    print(f"[monitor] Results dir: {RESULTS_DIR}")

    last_report_folds = -1
    last_report_epochs = -1

    while True:
        log_file = args.log_file or find_latest_log()
        if not log_file:
            print(f"[monitor] No log files found in {RESULTS_DIR}")
            if args.once:
                break
            time.sleep(args.interval)
            continue

        parsed = parse_log_file(log_file)
        gpu = get_gpu_status()

        # Determine if there's new info worth reporting
        current_folds = parsed['completed_folds']
        current_epochs = 0
        if parsed['in_progress_fold']:
            current_epochs = len(parsed['in_progress_fold']['epochs'])

        should_report = (
            current_folds != last_report_folds or
            current_epochs != last_report_epochs
        )

        if should_report:
            report = format_report(parsed, gpu)
            send_discord(report)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Report sent:")
            print(report)
            last_report_folds = current_folds
            last_report_epochs = current_epochs
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new progress. GPU: {gpu}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
