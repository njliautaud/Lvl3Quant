"""
Wider Hybrid WF experiment launcher.

Tests the WiderHybridModel (wider CNN + improved Transformer + engineered features)
using the walk-forward framework from train_walkforward.py.

Architecture:
  - Wider BookSpatialCNN (64,128,256,512) → 512-dim
  - Transformer (d_model=256, 8 heads, 3 layers) → 256-dim
  - Feature MLP (17 engineered features) → 64-dim
  - Fusion: [832] → 256 → 1
  - Expected: ~15-18M params

Strategy: monkey-patch train_walkforward.py to:
  1. Use 'hybrid' model type (to load both book + event data)
  2. Replace build_model() to return WiderHybridModel
  3. Replace _forward_batch() to pass engineered features
  4. Extend BarDataset to pre-compute and return engineered features
  5. Replace collate_hybrid() to include features in batch

Does NOT modify train_walkforward.py — all changes via monkey-patching.
"""

import sys
import os
import gc
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

import numpy as np
import torch
import torch.nn as nn

import train_walkforward as twf
from wider_hybrid_model import WiderHybridModel
from event_features import compute_event_features


# =============================================================================
# 1. Monkey-patch build_model to return WiderHybridModel for 'hybrid'
# =============================================================================

_original_build_model = twf.build_model

def patched_build_model(model_type, device, window_size=20, augment=False):
    """Build WiderHybridModel instead of the standard HybridModel."""
    if model_type == 'hybrid':
        model = WiderHybridModel(
            book_enc_dim=512,
            event_enc_dim=256,
            engineered_dim=17,
            eng_enc_dim=64,
            dropout=0.1,
            num_classes=1,
            window_size=window_size,
        )
        model = model.to(device)
        return model
    else:
        return _original_build_model(model_type, device, window_size, augment)

twf.build_model = patched_build_model


# =============================================================================
# 2. Monkey-patch _forward_batch to pass engineered features
# =============================================================================

def patched_forward_batch(model, batch, model_type, device):
    """Forward pass that includes engineered features for hybrid model."""
    if model_type == 'hybrid':
        # Our patched collate returns 5 items (with features)
        if len(batch) == 5:
            book_windows, event_seqs, event_lengths, eng_features, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            eng_features = eng_features.to(device)
            preds = model(book_windows, event_seqs, event_lengths, eng_features).squeeze(-1)
        else:
            # Fallback: standard 4-item batch (no features)
            book_windows, event_seqs, event_lengths, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            preds = model(book_windows, event_seqs, event_lengths).squeeze(-1)
        targets = targets.to(device)
        return preds, targets
    else:
        # Use original for non-hybrid
        return twf._original_forward_batch(model, batch, model_type, device)

# Save original before patching
twf._original_forward_batch = twf._forward_batch
twf._forward_batch = patched_forward_batch


# =============================================================================
# 3. Monkey-patch BarDataset to pre-compute engineered features
# =============================================================================

_OriginalBarDataset = twf.BarDataset

class PatchedBarDataset(_OriginalBarDataset):
    """Extended BarDataset that pre-computes engineered features for hybrid model."""

    def __init__(self, day_data_list, target, day_boundaries, model_type='hybrid',
                 window_size=20, horizon=20, subsample=1):
        # Call parent init (handles book tensors + event sequences)
        super().__init__(
            day_data_list, target, day_boundaries,
            model_type=model_type, window_size=window_size,
            horizon=horizon, subsample=subsample,
        )

        # Pre-compute engineered features for hybrid model
        if model_type == 'hybrid':
            # Concatenate event data across days
            all_event_seqs = []
            all_event_lens = []
            for d in day_data_list:
                if 'event_sequences' in d:
                    all_event_seqs.append(d['event_sequences'])
                    all_event_lens.append(d['sequence_lengths'])
                else:
                    n = len(d['book_tensors'])
                    all_event_seqs.append(np.zeros((n, 200, 5), dtype=np.int16))
                    all_event_lens.append(np.ones(n, dtype=np.uint16))

            event_seqs_concat = np.concatenate(all_event_seqs, axis=0)
            event_lens_concat = np.concatenate(all_event_lens, axis=0)

            # Compute all 17 features in bulk (fast, vectorized)
            self.eng_features = compute_event_features(
                event_seqs_concat, event_lens_concat
            ).astype(np.float32)

            # Z-score normalize features (per-feature, using training stats)
            self._feat_mean = self.eng_features.mean(axis=0)
            self._feat_std = self.eng_features.std(axis=0)
            self._feat_std = np.where(self._feat_std > 1e-8, self._feat_std, 1.0)
            self.eng_features = (self.eng_features - self._feat_mean) / self._feat_std
        else:
            self.eng_features = None

    def __getitem__(self, idx):
        base = super().__getitem__(idx)

        if self.model_type == 'hybrid' and self.eng_features is not None:
            i = int(self.valid_indices[idx])
            feats = torch.from_numpy(self.eng_features[i].copy())
            # base = (book_window, event_seq, event_len, target)
            # Return: (book_window, event_seq, event_len, features, target)
            return base[0], base[1], base[2], feats, base[3]

        return base

twf.BarDataset = PatchedBarDataset


# =============================================================================
# 4. Monkey-patch collate_hybrid to include engineered features
# =============================================================================

def patched_collate_hybrid(batch):
    """Collate that handles the extra engineered features tensor."""
    if len(batch[0]) == 5:
        # (book_window, event_seq, event_len, features, target)
        book_windows  = torch.stack([b[0] for b in batch])
        event_seqs    = torch.stack([b[1] for b in batch])
        event_lengths = torch.tensor([b[2] for b in batch])
        eng_features  = torch.stack([b[3] for b in batch])
        targets       = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return book_windows, event_seqs, event_lengths, eng_features, targets
    else:
        # Fallback to standard 4-item collate
        book_windows  = torch.stack([b[0] for b in batch])
        event_seqs    = torch.stack([b[1] for b in batch])
        event_lengths = torch.tensor([b[2] for b in batch])
        targets       = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return book_windows, event_seqs, event_lengths, targets

twf.collate_hybrid = patched_collate_hybrid


# =============================================================================
# 5. Override checkpoint naming to avoid conflicts
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'hybrid'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# No backup/restore dance needed — each model has its own directory


# =============================================================================
# Run
# =============================================================================

print("=" * 60)
print("WIDER HYBRID EXPERIMENT")
print("  CNN backbone:    spatial=(64,128,256,512) temporal=512 → 512d")
print("  Transformer:     d_model=256, 8 heads, 3 layers → 256d")
print("  Feature MLP:     17 engineered features → 64d")
print("  Fusion:          [832] → 256 → 1")
print("  Settings:        epochs=2, batch=256, subsample=5, max_days=15, expanding, warm_start=True")
print("=" * 60)

try:
    # Check wider hybrid checkpoint for resume fold (in isolated directory)
    _wider_h_ckpts = sorted(RESULTS_DIR.glob('checkpoint_wider_hybrid_*.json'))
    _wider_h_ckpts += sorted(RESULTS_DIR.glob('checkpoint_hybrid_*.json'))  # pre-rename
    _resume_fold_h = 0
    for _whc in _wider_h_ckpts:
        try:
            with open(_whc) as _whf:
                _whd = json.load(_whf)
            _nhf = _whd.get('completed_folds', 0)
            if _nhf > _resume_fold_h:
                _resume_fold_h = _nhf
                print(f"Found wider hybrid checkpoint: {_whc.name} with {_nhf} folds")
        except Exception:
            pass

    # Run walkforward using the 'hybrid' model type
    # (our patches redirect build_model, _forward_batch, BarDataset, and collate)
    sys.argv = [
        'run_wider_hybrid_wf.py',
        '--model', 'hybrid',
        '--epochs', '2',
        '--batch-size', '256',
        '--subsample-train', '5',
        '--max-train-days', '15',
        '--device', 'cuda',
        '--warm-start',
        '--output-dir', str(RESULTS_DIR),
    ]
    if _resume_fold_h > 0:
        sys.argv += ['--start-fold', str(_resume_fold_h)]
        print(f"Resuming from fold {_resume_fold_h} via --start-fold")
    twf.main()

finally:
    # With isolated directories, just rename hybrid→wider_hybrid in output names
    for ckpt in RESULTS_DIR.glob('checkpoint_hybrid_*.json'):
        wider_name = ckpt.name.replace('checkpoint_hybrid_', 'checkpoint_wider_hybrid_')
        ckpt.rename(ckpt.parent / wider_name)
        print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('oos_predictions_hybrid_*.npz'):
        wider_name = pred.name.replace('oos_predictions_hybrid_', 'oos_predictions_wider_hybrid_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
    for pred in RESULTS_DIR.glob('ckpt_preds_hybrid_*.npz'):
        wider_name = pred.name.replace('ckpt_preds_hybrid_', 'ckpt_preds_wider_hybrid_')
        if not (pred.parent / wider_name).exists():
            pred.rename(pred.parent / wider_name)
            print(f"Renamed: {wider_name}")
