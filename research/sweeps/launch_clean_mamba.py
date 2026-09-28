#!/usr/bin/env python3
import os
import subprocess
import sys

# Set environment variables
os.environ['EVENT_N_FOLDS'] = '2'
os.environ['EVENT_EPOCHS'] = '3'
os.environ['MAMBA_D_MODEL'] = '128'
os.environ['MAMBA_N_LAYERS'] = '4'
os.environ['MAMBA_D_STATE'] = '64'
os.environ['EVENT_BATCH_SIZE'] = '64'
os.environ['EVENT_WINDOW_SIZE'] = '1000'
os.environ['EVENT_STRIDE'] = '500'

# Change to Lvl3Quant directory on Neptune
os.chdir('/home/nick/Lvl3Quant')

# Activate virtual environment and run training
cmd = [
    '/home/nick/training-env/bin/python', '-u',
    'alpha_discovery/deep_models/train_event_mamba.py',
    '--data-dir', '/tmp/mbo_clean_neptune',
    '--output-dir', f'/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/mamba_clean_baseline_{os.popen("date +%Y%m%d_%H%M").read().strip()}'
]

sys.exit(subprocess.call(cmd))
