"""
Launch mfe_mae_cnn.py walk-forward experiment on Uranus.

MFE/MAE model: predicts Max Favorable Excursion and Max Adverse Excursion.
Entry signal: mfe_pred > 2.0 * mae_pred (regardless of direction).
Pure regression, no classification head, no NaN collapse risk.

Usage (run on Uranus):
    python launch_mfe_mae_wf.py
"""
import os
import subprocess
import sys
from pathlib import Path

# Uranus paths
LVL3_ROOT = Path(r'C:\Users\nick\Lvl3Quant')
SCRIPT = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'mfe_mae_cnn.py'
DATA_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache'
OUTPUT_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'mfe_mae_wf'

CMD = [
    sys.executable, str(SCRIPT),
    '--data_dir', str(DATA_DIR),
    '--output_dir', str(OUTPUT_DIR),
    '--min_train_days', '5',
    '--max_train_days', 'None',    # expanding window
    '--epochs', '2',
    '--batch_size', '256',
    '--lr', '5e-4',
    '--ratio_threshold', '2.0',
    '--horizon', '20',
    '--subsample', '10',
    '--num_workers', '8',
    '--device', 'cuda',
    '--checkpoint_interval', '5',
]

# Remove 'None' string arg - argparse needs it absent for default=None
CMD = [c for c in CMD if c != 'None']
# Re-add without --max_train_days (let it default to None = expanding)
CMD = [c for c in CMD if c not in ('--max_train_days',)]

print('Launching MFE/MAE Prediction CNN WF experiment on Uranus...')
print('Architecture: BookSpatialCNN wider (64,128,256,512), ~12.6M params')
print('Objective: MSE(MFE) + MSE(MAE), entry when mfe_pred > 2.0 * mae_pred')
print('Command:', ' '.join(CMD))
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUTPUT_DIR / 'mfe_mae_wf.log'
err_file = OUTPUT_DIR / 'mfe_mae_wf_err.log'

with open(log_file, 'w') as lf, open(err_file, 'w') as ef:
    proc = subprocess.Popen(CMD, stdout=lf, stderr=ef)
    print(f'PID: {proc.pid}')
    print(f'Log: {log_file}')
    print(f'Err: {err_file}')

pid_file = OUTPUT_DIR / 'mfe_mae_wf.pid'
pid_file.write_text(str(proc.pid))
print(f'PID file: {pid_file}')
