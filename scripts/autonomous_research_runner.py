#!/usr/bin/env python3
"""
Autonomous Research Runner
===========================
Runs queued research experiments completely independently of the Claude session.
Each experiment is a self-contained Python script that:
1. Downloads data
2. Runs strategy backtest
3. Runs adversarial validation (permutation, sub-period, regime)
4. Logs results to a shared results file
5. Moves to the next experiment

This script runs via cron or PM2 and survives Claude session restarts.

Usage:
    python3 autonomous_research_runner.py           # Run next queued experiment
    python3 autonomous_research_runner.py --status   # Show queue status
    python3 autonomous_research_runner.py --add "name" "description"  # Add to queue
"""

import json
import os
import sys
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

BASE_DIR = Path('/home/jupiter/Lvl3Quant')
QUEUE_FILE = BASE_DIR / 'data/research_queue.json'
RESULTS_FILE = BASE_DIR / 'data/research_results.json'
LOG_DIR = BASE_DIR / 'logs/autonomous_research'
SCRIPTS_DIR = BASE_DIR / 'scripts/growth_research/autonomous'

LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_queue():
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            return json.load(f)
    return []


def save_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    log_file = LOG_DIR / f"runner_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'a') as f:
        f.write(line + '\n')


def send_discord_alert(msg):
    """Send alert via webhook if available"""
    webhook_script = Path('/home/jupiter/teleclaude-main/utils/webhook_notifier.js')
    if webhook_script.exists():
        try:
            subprocess.run(['node', str(webhook_script), msg],
                          capture_output=True, timeout=10)
        except:
            pass


def run_experiment(experiment):
    """Run a single experiment script and capture results"""
    script_path = experiment.get('script')
    if not script_path or not os.path.exists(script_path):
        return {'status': 'error', 'error': f'Script not found: {script_path}'}

    log(f"Starting experiment: {experiment['name']}")
    log(f"Script: {script_path}")

    log_file = LOG_DIR / f"{experiment['name'].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    start_time = time.time()
    try:
        result = subprocess.run(
            ['python3', '-u', script_path],
            capture_output=True,
            text=True,
            timeout=experiment.get('timeout', 3600),  # Default 1hr timeout
            cwd=str(BASE_DIR)
        )

        elapsed = time.time() - start_time

        # Save full log
        with open(log_file, 'w') as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}")

        # Parse results from stdout (scripts should print JSON summary at end)
        output = result.stdout
        summary = None

        # Try to find JSON summary in output (between RESULT_JSON markers)
        if '<<<RESULT_JSON>>>' in output and '<<<END_RESULT_JSON>>>' in output:
            json_str = output.split('<<<RESULT_JSON>>>')[1].split('<<<END_RESULT_JSON>>>')[0].strip()
            try:
                summary = json.loads(json_str)
            except:
                pass

        return {
            'status': 'completed' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'elapsed_seconds': elapsed,
            'log_file': str(log_file),
            'summary': summary,
            'last_lines': output.strip().split('\n')[-20:] if output else [],
            'stderr_tail': result.stderr.strip().split('\n')[-5:] if result.stderr else []
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log(f"TIMEOUT after {elapsed:.0f}s")
        return {'status': 'timeout', 'elapsed_seconds': elapsed, 'log_file': str(log_file)}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}


def process_queue():
    """Process the next experiment in the queue"""
    queue = load_queue()

    if not queue:
        log("Queue empty — nothing to run")
        return

    # Find next pending experiment
    pending = [e for e in queue if e.get('status') == 'pending']
    if not pending:
        log("No pending experiments in queue")
        return

    experiment = pending[0]
    experiment['status'] = 'running'
    experiment['started_at'] = datetime.now().isoformat()
    save_queue(queue)

    # Run it
    result = run_experiment(experiment)

    # Update queue
    experiment['status'] = result['status']
    experiment['completed_at'] = datetime.now().isoformat()
    experiment['result'] = result
    save_queue(queue)

    # Save to results file
    results = load_results()
    results.append({
        'name': experiment['name'],
        'description': experiment.get('description', ''),
        'status': result['status'],
        'started_at': experiment['started_at'],
        'completed_at': experiment['completed_at'],
        'elapsed_seconds': result.get('elapsed_seconds', 0),
        'summary': result.get('summary'),
        'last_lines': result.get('last_lines', []),
    })
    save_results(results)

    # Log and alert
    elapsed = result.get('elapsed_seconds', 0)
    status = result['status']
    log(f"Experiment '{experiment['name']}' {status} in {elapsed:.0f}s")

    if result.get('summary'):
        s = result['summary']
        log(f"  Sharpe: {s.get('sharpe', '?')}, CAGR: {s.get('cagr', '?')}, "
            f"MaxDD: {s.get('max_dd', '?')}, Perm: {s.get('perm_pass', '?')}")

    # Check if more experiments pending
    remaining = len([e for e in queue if e.get('status') == 'pending'])
    log(f"Queue: {remaining} experiments remaining")

    # If more pending, run next
    if remaining > 0:
        log("Running next experiment...")
        process_queue()


def show_status():
    """Show queue status"""
    queue = load_queue()
    results = load_results()

    print(f"\n{'='*60}")
    print(f"RESEARCH QUEUE STATUS")
    print(f"{'='*60}")

    if not queue:
        print("  Queue is empty")
    else:
        for e in queue:
            status = e.get('status', '?')
            icon = {'pending': '⏳', 'running': '🔄', 'completed': '✅', 'failed': '❌', 'timeout': '⏰'}.get(status, '?')
            print(f"  {icon} [{status}] {e['name']}")

    print(f"\n{'='*60}")
    print(f"COMPLETED RESULTS ({len(results)} total)")
    print(f"{'='*60}")

    for r in results[-10:]:
        s = r.get('summary', {})
        sharpe = s.get('sharpe', '?') if s else '?'
        perm = s.get('perm_pass', '?') if s else '?'
        print(f"  {r['status']}: {r['name']} — Sharpe {sharpe}, Perm {perm}")


def add_to_queue(name, description, script_path, timeout=3600):
    """Add experiment to queue"""
    queue = load_queue()
    queue.append({
        'name': name,
        'description': description,
        'script': script_path,
        'timeout': timeout,
        'status': 'pending',
        'added_at': datetime.now().isoformat()
    })
    save_queue(queue)
    log(f"Added to queue: {name}")


if __name__ == '__main__':
    if '--status' in sys.argv:
        show_status()
    elif '--add' in sys.argv:
        idx = sys.argv.index('--add')
        name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'unnamed'
        desc = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else ''
        script = sys.argv[idx + 3] if idx + 3 < len(sys.argv) else ''
        add_to_queue(name, desc, script)
    else:
        process_queue()
