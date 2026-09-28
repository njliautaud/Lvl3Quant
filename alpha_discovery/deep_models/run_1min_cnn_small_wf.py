"""
1-minute horizon SMALL BookSpatialCNN walk-forward experiment.

Anti-overfit design:
  - Smaller model: (16, 32, 64, 128) + temporal=128 = ~1M params
  - Higher dropout: 0.4 (vs standard 0.1)
  - Higher subsample: 10x (vs 5x) — more diverse training samples
  - More regularization: weight_decay via optimizer

Bar interval: 100ms => 1min = 600 bars horizon.
Target: mfe_net_1min — Max Favorable Excursion net over 60-second forward window.

Previous run (standard 4M model): IC 0.046-0.094 but val_loss 4x train_loss (severe overfit).
This run tests whether a smaller model can maintain IC while reducing overfitting.
"""
import sys
import os
from pathlib import Path

# Set thread limits BEFORE importing
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'alpha_discovery' / 'deep_models'))
sys.path.insert(0, str(PROJECT_ROOT))

import book_spatial_cnn
import train_walkforward as twf

# ── Monkey-patch to smaller architecture ──────────────────────────────────────

OriginalCNN = book_spatial_cnn.BookSpatialCNN

class SmallBookSpatialCNN(OriginalCNN):
    """Smaller BookSpatialCNN for 1-min horizon (anti-overfit)."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (16, 32, 64, 128)
        kwargs['temporal_channels'] = 128
        kwargs['dropout'] = 0.4
        super().__init__(**kwargs)

# Replace globally
book_spatial_cnn.BookSpatialCNN = SmallBookSpatialCNN
twf.BookSpatialCNN = SmallBookSpatialCNN

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'cnn_wf_1min_small'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1-MINUTE HORIZON SMALL BookSpatialCNN (anti-overfit)")
print("  Architecture: (16, 32, 64, 128) + temporal=128")
print("  Expected: ~1M params (vs 4M standard)")
print("  Dropout: 0.4 (vs 0.1 standard)")
print("  Subsample: 10x (vs 5x)")
print("  Horizon: 600 bars = 60s @ 100ms bar interval")
print(f"  Output: {RESULTS_DIR}")
print("=" * 60)

# Run walkforward with 1-minute horizon
sys.argv = [
    'run_1min_cnn_small_wf.py',
    '--model', 'book',
    '--epochs', '3',
    '--batch-size', '512',
    '--subsample-train', '10',
    '--max-train-days', '15',
    '--horizon-bars', '600',
    '--window-size', '20',
    '--lr', '3e-4',
    '--device', 'cuda',
    '--warm-start',
    '--output-dir', str(RESULTS_DIR),
]

twf.main()
