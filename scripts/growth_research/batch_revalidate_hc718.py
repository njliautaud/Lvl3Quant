#!/usr/bin/env python3
"""
HC #718 R5 — Batch re-validation of all 7 ML strategies after fixing:
  1. Label gap (train_end = test_start - LABEL_HORIZON)
  2. Permutation tests (shuffle signals, not returns)
  3. Transaction costs

Runs each script as a subprocess, captures output, and produces summary.
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

SCRIPTS = [
    'ml_currency_carry.py',
    'ml_commodity_trend.py',
    'ml_tail_risk_hedging.py',
    'ml_stat_arb.py',
    'ml_vol_breakout.py',
    'ml_trend_following_v2.py',
    'ml_sector_momentum_v1.py',
    'ml_bond_duration_timing.py',
]

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/hc718_revalidation')
OUTPUT_DIR.mkdir(exist_ok=True)

results = {}

for script in SCRIPTS:
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        print(f"\n{'='*60}")
        print(f"SKIP: {script} — file not found")
        results[script] = {'status': 'SKIP', 'reason': 'file not found'}
        continue

    name = script.replace('.py', '').replace('ml_', '')
    print(f"\n{'='*60}")
    print(f"RUNNING: {script} ({time.strftime('%H:%M:%S')})")
    print(f"{'='*60}")

    log_path = OUTPUT_DIR / f"{name}_output.txt"
    json_path = OUTPUT_DIR / f"{name}_results.json"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True,
            timeout=600,  # 10 min max per strategy
            cwd=str(SCRIPT_DIR)
        )
        elapsed = time.time() - t0

        output = result.stdout + '\n' + result.stderr
        log_path.write_text(output)

        # Parse key metrics from output
        metrics = {
            'status': 'OK' if result.returncode == 0 else 'ERROR',
            'returncode': result.returncode,
            'elapsed_s': round(elapsed, 1),
        }

        # Extract key lines
        for line in output.split('\n'):
            line_lower = line.lower().strip()
            if 'sharpe' in line_lower and ('observed' in line_lower or 'annualized' in line_lower or 'ml sharpe' in line_lower):
                metrics['sharpe_line'] = line.strip()
            if 'p_value' in line_lower or 'p=' in line_lower or 'p value' in line_lower:
                if 'perm' in line_lower or 'permut' in line_lower:
                    metrics['perm_line'] = line.strip()
            if 'verdict' in line_lower and ('pass' in line_lower or 'fail' in line_lower):
                metrics.setdefault('verdicts', []).append(line.strip())
            if 'final verdict' in line_lower or 'overall' in line_lower:
                metrics['final_line'] = line.strip()
            if 'cagr' in line_lower:
                metrics['cagr_line'] = line.strip()

        # Check for JSON output file from the strategy itself
        for f in OUTPUT_DIR.parent.glob(f"*{name}*results*.json"):
            try:
                with open(f) as fh:
                    strategy_json = json.load(fh)
                    metrics['strategy_output'] = str(f)
            except:
                pass

        results[script] = metrics
        print(f"  Done in {elapsed:.1f}s, exit={result.returncode}")

        # Print last 20 lines of output
        lines = [l for l in output.split('\n') if l.strip()]
        for l in lines[-20:]:
            print(f"  {l}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s")
        results[script] = {'status': 'TIMEOUT', 'elapsed_s': round(elapsed, 1)}
    except Exception as e:
        print(f"  ERROR: {e}")
        results[script] = {'status': 'ERROR', 'error': str(e)}

# Summary
print(f"\n\n{'='*60}")
print("HC #718 R5 REVALIDATION SUMMARY")
print(f"{'='*60}")

for script, m in results.items():
    name = script.replace('.py', '').replace('ml_', '')
    status = m.get('status', '?')
    sharpe = m.get('sharpe_line', 'N/A')
    perm = m.get('perm_line', 'N/A')
    verdicts = m.get('verdicts', [])
    final = m.get('final_line', '')

    print(f"\n{name}:")
    print(f"  Status: {status}")
    if sharpe != 'N/A':
        print(f"  {sharpe}")
    if perm != 'N/A':
        print(f"  {perm}")
    for v in verdicts[-3:]:
        print(f"  {v}")
    if final:
        print(f"  {final}")

# Save summary
summary_path = OUTPUT_DIR / 'batch_summary.json'
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to {summary_path}")
