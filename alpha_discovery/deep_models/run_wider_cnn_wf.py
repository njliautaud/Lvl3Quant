"""
Wider CNN WF experiment for Neptune GPU.
Tests BookSpatialCNN with doubled channel widths.
Standard: (32, 64, 128, 256) + temporal=256 = ~4M params
Wider:    (64, 128, 256, 512) + temporal=512 = ~16M params

Memory-safe: Only uses book_tensors (~1-2GB for 15 days of data).
Saves to separate checkpoint file so it doesn't conflict with standard CNN.
"""
import sys
import os
import json
import time
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

import book_spatial_cnn
import train_walkforward as twf

# ── Monkey-patch to wider architecture ──────────────────────────────────────

OriginalCNN = book_spatial_cnn.BookSpatialCNN

class WiderBookSpatialCNN(OriginalCNN):
    """2x wider BookSpatialCNN for architecture search."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)

# Replace globally
book_spatial_cnn.BookSpatialCNN = WiderBookSpatialCNN
twf.BookSpatialCNN = WiderBookSpatialCNN

# ── Monkey-patch checkpoint save to tag with wider_cnn marker ────────────────
_orig_json_dump = json.dump
def _tagged_json_dump(obj, fp, **kwargs):
    """Tag checkpoint files with wider_cnn=True for resume detection."""
    if isinstance(obj, dict) and 'completed_folds' in obj and 'fold_ics' in obj:
        obj['wider_cnn'] = True
    return _orig_json_dump(obj, fp, **kwargs)
json.dump = _tagged_json_dump

# ── Override checkpoint name to avoid conflicts ─────────────────────────────

RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# No backup/restore dance needed — each model has its own directory

print("=" * 60)
print("WIDER CNN EXPERIMENT")
print("  Architecture: spatial=(64,128,256,512) temporal=512")
print("  Expected: ~16M params (4x standard 4M)")
print("  Settings: epochs=2, batch=256, subsample=5, max_days=15, warm_start=True")
print("=" * 60)

try:
    # Run walkforward
    # Check wider CNN checkpoint for resume fold (in isolated directory)
    _wider_ckpts = sorted(RESULTS_DIR.glob('checkpoint_wider_cnn_*.json'))
    # Also check book-named checkpoints (pre-rename, same directory)
    _wider_ckpts += sorted(RESULTS_DIR.glob('checkpoint_book_*.json'))
    _resume_fold = 0
    for _wc in _wider_ckpts:
        try:
            with open(_wc) as _wf:
                _wd = json.load(_wf)
            _nf = _wd.get('completed_folds', 0)
            if _nf > _resume_fold:
                _resume_fold = _nf
                print(f"Found wider CNN checkpoint: {_wc.name} with {_nf} folds")
        except Exception:
            pass

    sys.argv = [
        'run_wider_cnn_wf.py',
        '--model', 'book',
        '--epochs', '2',
        '--batch-size', '256',
        '--subsample-train', '5',
        '--max-train-days', '15',
        '--device', 'cuda',
        '--warm-start',
        '--output-dir', str(RESULTS_DIR),
    ]
    if _resume_fold > 0:
        sys.argv += ['--start-fold', str(_resume_fold)]
        print(f"Resuming from fold {_resume_fold} via --start-fold")
    twf.main()

finally:
    # With isolated directories, just rename book→wider_cnn in output names
    for ckpt in RESULTS_DIR.glob('checkpoint_book_*.json'):
        wider_name = ckpt.name.replace('checkpoint_book_', 'checkpoint_wider_cnn_')
        ckpt.rename(ckpt.parent / wider_name)
        print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('oos_predictions_book_*.npz'):
        wider_name = pred.name.replace('oos_predictions_book_', 'oos_predictions_wider_cnn_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('ckpt_preds_book_*.npz'):
        wider_name = pred.name.replace('ckpt_preds_book_', 'ckpt_preds_wider_cnn_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
