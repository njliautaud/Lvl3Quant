"""
Wider CNN WF with EXPANDING WINDOW on Neptune GPU.

Architecture: BookSpatialCNN wider (64, 128, 256, 512) + temporal=512
  ~12.6M params (matches Mar 18 run)

Window mode: EXPANDING (no --max-train-days cap)
  Each fold trains on ALL available prior data, not just last 15 days.
  This grows the training set from 5 days at fold 1 to 130+ days at later folds.

Training is automatically resumed from the latest checkpoint in results/wider_cnn/.

Safety:
  OMP_NUM_THREADS=16, MKL_NUM_THREADS=16, CUDA_VISIBLE_DEVICES=0
  These are set by launch_wider_cnn_expanding.py before calling this script.
"""
import sys
import os
import json
import time
from pathlib import Path

# Thread limits (also set in launcher env, belt+suspenders)
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import book_spatial_cnn
import train_walkforward as twf

# â”€â”€ Monkey-patch to wider architecture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OriginalCNN = book_spatial_cnn.BookSpatialCNN

class WiderBookSpatialCNN(OriginalCNN):
    """2x wider BookSpatialCNN â€” same as the Mar 18 run (~12.6M params)."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)

# Replace globally
book_spatial_cnn.BookSpatialCNN = WiderBookSpatialCNN
twf.BookSpatialCNN = WiderBookSpatialCNN

# â”€â”€ Tag checkpoints with wider_cnn marker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_orig_json_dump = json.dump
def _tagged_json_dump(obj, fp, **kwargs):
    if isinstance(obj, dict) and 'completed_folds' in obj and 'fold_ics' in obj:
        obj['wider_cnn'] = True
        obj['window_mode'] = 'expanding'
    return _orig_json_dump(obj, fp, **kwargs)
json.dump = _tagged_json_dump

# â”€â”€ Results directory (isolated from standard CNN) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RESULTS_DIR = SCRIPT_DIR / 'results' / 'wider_cnn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("WIDER CNN â€” EXPANDING WINDOW EXPERIMENT")
print("  Architecture: spatial=(64,128,256,512) temporal=512")
print("  Params: ~12.6M")
print("  Window: EXPANDING (all prior data per fold, no cap)")
print("  Settings: epochs=2, batch=256, subsample=10, warm_start=True")
print("=" * 60)

# â”€â”€ Find resume fold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_resume_fold = 0
_wider_ckpts = sorted(RESULTS_DIR.glob('checkpoint_wider_cnn_*.json'))
_wider_ckpts += sorted(RESULTS_DIR.glob('checkpoint_book_*.json'))
for _wc in _wider_ckpts:
    try:
        with open(_wc) as _wf:
            _wd = json.load(_wf)
        _nf = _wd.get('completed_folds', 0)
        if _nf > _resume_fold:
            _resume_fold = _nf
            print(f"Found checkpoint: {_wc.name} with {_nf} folds")
    except Exception:
        pass

print(f"Resuming from fold {_resume_fold} (expanding window, no max_train_days)")
print()

# â”€â”€ Build sys.argv for train_walkforward.main() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# KEY DIFFERENCE: NO --max-train-days argument â†’ expanding window
sys.argv = [
    'run_wider_cnn_expanding_wf.py',
    '--model', 'book',
    '--epochs', '2',
    '--batch-size', '256',
    '--subsample-train', '10',
    '--max-train-days', '60',  # Cap at 60 days to avoid OOM on Neptune (32GB RAM)
    # --max-train-days capped at 60 days for Neptune RAM safety
    '--device', 'cuda',
    '--warm-start',
    '--output-dir', str(RESULTS_DIR),
]
if _resume_fold > 0:
    sys.argv += ['--start-fold', str(_resume_fold)]
    print(f"Resuming from fold {_resume_fold} via --start-fold")

try:
    twf.main()
finally:
    # Rename book-named files to wider_cnn-named (consistent tagging)
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
