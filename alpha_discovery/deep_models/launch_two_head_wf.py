"""
Launch two_head_cnn.py walk-forward experiment on Uranus.
Run this locally on Uranus (or via SSH) once E4 is confirmed idle.

Usage:
    python launch_two_head_wf.py
"""
import os
import subprocess
import sys
from pathlib import Path

# Uranus paths
LVL3_ROOT = Path(r'C:\Users\nick\Lvl3Quant')
SCRIPT = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'two_head_cnn.py'
DATA_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache'
OUTPUT_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'two_head_wf'

CMD = [
    sys.executable, str(SCRIPT),
    '--wf',
    '--data_dir', str(DATA_DIR),
    '--output_dir', str(OUTPUT_DIR),
    '--min_train_days', '5',
    '--epochs', '2',
    '--batch_size', '256',
    '--lr', '5e-4',
    '--alpha', '0.5',
    '--direction_threshold', '0.5',
    '--subsample', '10',
    '--num_workers', '8',
    '--device', 'cuda',
]

print('Launching Two-Head CNN WF experiment...')
print('Command:', ' '.join(CMD))
print()

# Run in background with output redirected to log file
log_file = OUTPUT_DIR / 'two_head_wf.log'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(log_file, 'w') as lf:
    proc = subprocess.Popen(CMD, stdout=lf, stderr=lf)
    print(f'PID: {proc.pid}')
    print(f'Log: {log_file}')

# Save PID
pid_file = OUTPUT_DIR / 'two_head_wf.pid'
pid_file.write_text(str(proc.pid))
print(f'PID file: {pid_file}')
