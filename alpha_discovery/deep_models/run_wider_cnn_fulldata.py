"""
WIDER CNN — FULL-DATA SINGLE-PASS TRAINING on Neptune 3090.

Purpose: Train wider CNN on ALL available data in one shot to get a deployable
model fast, instead of doing 168-fold walkforward.

Architecture: BookSpatialCNN wider (64, 128, 256, 512) + temporal=512
  ~12.6M params

Strategy: Use walkforward with min_train_days=170 so there's effectively 1 fold
that trains on nearly all available data. More epochs (30) since we're doing
a single training pass instead of 2-3 epochs x 168 folds.

No warm-start (fresh weights for clean full-data model).
No max-train-days (expanding window = all data).

Output: results/wider_cnn_fulldata/
"""
import sys
import os
import json
import time
from pathlib import Path

# Thread limits
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import book_spatial_cnn
import train_walkforward as twf

# ── Monkey-patch to wider architecture ──────────────────────────────────────
OriginalCNN = book_spatial_cnn.BookSpatialCNN

class WiderBookSpatialCNN(OriginalCNN):
    """2x wider BookSpatialCNN — same as the Mar 18 run (~12.6M params)."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)

# Replace globally
book_spatial_cnn.BookSpatialCNN = WiderBookSpatialCNN
twf.BookSpatialCNN = WiderBookSpatialCNN

# ── Tag checkpoints with wider_cnn_fulldata marker ──────────────────────────
_orig_json_dump = json.dump
def _tagged_json_dump(obj, fp, **kwargs):
    if isinstance(obj, dict) and 'completed_folds' in obj and 'fold_ics' in obj:
        obj['wider_cnn'] = True
        obj['window_mode'] = 'fulldata'
        obj['training_mode'] = 'single_pass'
    return _orig_json_dump(obj, fp, **kwargs)
json.dump = _tagged_json_dump

# ── Results directory (separate from walkforward results) ───────────────────
RESULTS_DIR = SCRIPT_DIR / 'results' / 'wider_cnn_fulldata'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("WIDER CNN — FULL-DATA SINGLE-PASS TRAINING")
print("  Architecture: spatial=(64,128,256,512) temporal=512")
print("  Params: ~12.6M")
print("  Window: EXPANDING (all data, min_train_days=170)")
print("  Epochs: 30 (single pass, more epochs for convergence)")
print("  Batch: 512, subsample=3, NO warm-start")
print("  Output:", RESULTS_DIR)
print("=" * 60)

# ── Build sys.argv for train_walkforward.main() ─────────────────────────────
sys.argv = [
    'run_wider_cnn_fulldata.py',
    '--model', 'book',
    '--epochs', '30',
    '--batch-size', '512',
    '--subsample-train', '3',
    '--min-train-days', '170',   # With 173 days total, this creates ~2 folds max
    '--device', 'cuda',
    # NO --warm-start (fresh weights)
    # NO --max-train-days (expanding = all data)
    '--output-dir', str(RESULTS_DIR),
]

print(f"\nLaunching train_walkforward.main() with args:")
print(f"  {' '.join(sys.argv[1:])}")
print()

try:
    twf.main()
except Exception as e:
    print(f"\nTRAINING FAILED: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Rename book-named files to wider_cnn_fulldata-named
    for ckpt in RESULTS_DIR.glob('checkpoint_book_*.json'):
        wider_name = ckpt.name.replace('checkpoint_book_', 'checkpoint_wider_cnn_fulldata_')
        ckpt.rename(ckpt.parent / wider_name)
        print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('oos_predictions_book_*.npz'):
        wider_name = pred.name.replace('oos_predictions_book_', 'oos_predictions_wider_cnn_fulldata_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
    for wt in RESULTS_DIR.glob('latest.pt'):
        import shutil
        dest = RESULTS_DIR / 'wider_cnn_fulldata_final.pt'
        shutil.copy2(str(wt), str(dest))
        print(f"Copied final weights: {dest}")
    print("\nTraining complete. Check results in:", RESULTS_DIR)
