"""
1-minute horizon BookSpatialCNN walk-forward experiment.

Bar interval: 100ms => 1min = 600 bars horizon.
Standard architecture: (32, 64, 128, 256) + temporal=256 = ~4M params (fits 3070 8GB).
Target: mfe_net_1min — Max Favorable Excursion net over 60-second forward window.

Saves results to results/cnn_wf_1min/ to avoid conflicts with other horizon runs.
"""
import sys
import os
from pathlib import Path

# Set thread limits BEFORE importing
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'alpha_discovery' / 'deep_models'))
sys.path.insert(0, str(PROJECT_ROOT))

import train_walkforward as twf

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'cnn_wf_1min'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1-MINUTE HORIZON BookSpatialCNN WALK-FORWARD")
print("  Architecture: standard (32, 64, 128, 256) + temporal=256")
print("  Params: ~4M (fits 3070 8GB)")
print("  Horizon: 600 bars = 60s @ 100ms bar interval")
print("  Settings: epochs=3, batch=512, subsample=5, lr=3e-4")
print(f"  Output: {RESULTS_DIR}")
print("=" * 60)

# Run walkforward with 1-minute horizon
sys.argv = [
    'run_1min_cnn_wf.py',
    '--model', 'book',
    '--epochs', '3',
    '--batch-size', '512',
    '--subsample-train', '5',
    '--max-train-days', '15',
    '--horizon-bars', '600',       # 600 bars * 100ms = 60 seconds
    '--window-size', '20',
    '--lr', '3e-4',
    '--device', 'cuda',
    '--output-dir', str(RESULTS_DIR),
]

twf.main()
