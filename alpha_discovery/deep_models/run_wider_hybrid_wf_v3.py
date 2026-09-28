"""
Wider Hybrid WF v3 - Optimized for generalization (anti-overfit).

Changes from v2:
  - Model size: ~15.5M -> ~6-8M params (halved spatial channels)
    - Spatial channels: (64,128,256,512) -> (32,64,128,256)
    - Temporal channels: 512 -> 256
    - book_enc_dim: 512 -> 256
    - Fusion: [256+256+64]=576 -> 128 -> 1
  - Sliding window: 30 days max (drop stale data)
  - Subsample: 5x -> 8x (reduce training overfitting)
  - Epochs: 6 -> 2 (warm start)
  - Dropout: 0.4 -> 0.5
  - Weight decay: 5e-3 -> 1e-2
  - Learning rate: 1e-4 -> 5e-5
  - Gradient clipping: 0.5 -> 0.3
  - Batch size: 512 (same)

All changes applied via monkey-patching - does NOT modify train_walkforward.py.
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
from typing import Optional, Tuple
from torch.utils.data import DataLoader

import train_walkforward as twf
from event_features import compute_event_features
from book_spatial_cnn import BookSpatialCNN
from event_transformer import EventTransformer


# =============================================================================
# 1. Custom smaller hybrid model (halved CNN channels)
# =============================================================================

class SmallerHybridModel(nn.Module):
    """
    Halved version of WiderHybridModel for better generalization.

    CNN: spatial=(32,64,128,256) temporal=256 -> 256d  (was 64,128,256,512 -> 512d)
    Transformer: d_model=256, 8 heads, 3 layers -> 256d (same)
    Feature MLP: 17 -> 64d (same)
    Fusion: [256+256+64]=576 -> 128 -> 1
    """
    def __init__(self, dropout=0.5, window_size=20):
        super().__init__()

        # Halved CNN backbone: (32,64,128,256) instead of (64,128,256,512)
        self.book_encoder = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=(32, 64, 128, 256),   # HALVED from (64,128,256,512)
            temporal_channels=256,                  # HALVED from 512
            dropout=dropout,
            num_classes=256,                        # output dim (was 512)
        )

        # Same transformer as v2
        self.event_encoder = EventTransformer(
            d_model=256,
            nhead=8,
            num_layers=3,
            dim_feedforward=1024,
            dropout=dropout,
            num_classes=256,
            use_cnn_stem=True,
        )

        # Same feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # Smaller fusion head: 576 -> 128 -> 1
        fusion_dim = 256 + 256 + 64  # = 576
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),    # was 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, book_windows, event_seqs, event_lengths, engineered_features=None):
        book_feats = self.book_encoder(book_windows)
        event_feats = self.event_encoder(event_seqs, event_lengths)

        if engineered_features is not None:
            feat_feats = self.feature_mlp(engineered_features)
        else:
            feat_feats = torch.zeros(
                book_feats.shape[0], 64,
                device=book_feats.device, dtype=book_feats.dtype,
            )

        fused = torch.cat([book_feats, event_feats, feat_feats], dim=-1)
        fused = self.fusion_norm(fused)
        return self.fusion_head(fused)


_original_build_model = twf.build_model

def patched_build_model(model_type, device, window_size=20, augment=False):
    """Build SmallerHybridModel with halved channels and higher dropout."""
    if model_type == 'hybrid':
        model = SmallerHybridModel(dropout=0.5, window_size=window_size)
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
        if len(batch) == 5:
            book_windows, event_seqs, event_lengths, eng_features, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            eng_features = eng_features.to(device)
            preds = model(book_windows, event_seqs, event_lengths, eng_features).squeeze(-1)
        else:
            book_windows, event_seqs, event_lengths, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            preds = model(book_windows, event_seqs, event_lengths).squeeze(-1)
        targets = targets.to(device)
        return preds, targets
    else:
        return twf._original_forward_batch(model, batch, model_type, device)

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
        super().__init__(
            day_data_list, target, day_boundaries,
            model_type=model_type, window_size=window_size,
            horizon=horizon, subsample=subsample,
        )

        if model_type == 'hybrid':
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

            self.eng_features = compute_event_features(
                event_seqs_concat, event_lens_concat
            ).astype(np.float32)

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
            return base[0], base[1], base[2], feats, base[3]

        return base

twf.BarDataset = PatchedBarDataset


# =============================================================================
# 4. Monkey-patch collate_hybrid to include engineered features
# =============================================================================

def patched_collate_hybrid(batch):
    """Collate that handles the extra engineered features tensor."""
    if len(batch[0]) == 5:
        book_windows  = torch.stack([b[0] for b in batch])
        event_seqs    = torch.stack([b[1] for b in batch])
        event_lengths = torch.tensor([b[2] for b in batch])
        eng_features  = torch.stack([b[3] for b in batch])
        targets       = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return book_windows, event_seqs, event_lengths, eng_features, targets
    else:
        book_windows  = torch.stack([b[0] for b in batch])
        event_seqs    = torch.stack([b[1] for b in batch])
        event_lengths = torch.tensor([b[2] for b in batch])
        targets       = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return book_windows, event_seqs, event_lengths, targets

twf.collate_hybrid = patched_collate_hybrid


# =============================================================================
# 5. Monkey-patch train_epoch for tighter gradient clipping (0.3)
# =============================================================================

def patched_train_epoch(model, loader, optimizer, device, model_type, scaler=None):
    """Train epoch with tighter grad clipping (0.3)."""
    model.train()
    total_loss = 0.0
    total_n    = 0
    criterion  = nn.HuberLoss(delta=1.0)

    for batch in loader:
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds, targets = twf._forward_batch(model, batch, model_type, device)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.3)   # WAS 0.5 in v2, 1.0 in base
            scaler.step(optimizer)
            scaler.update()
        else:
            preds, targets = twf._forward_batch(model, batch, model_type, device)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()

        n = targets.shape[0]
        total_loss += loss.item() * n
        total_n    += n

    return total_loss / max(total_n, 1), total_n

twf.train_epoch = patched_train_epoch


# =============================================================================
# 6. Override checkpoint naming to avoid conflicts
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'hybrid'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DISABLED: Don't back up checkpoints -- we need them for resume
# hybrid_ckpts = list(RESULTS_DIR.glob('checkpoint_hybrid_*.json'))
renamed = []
print("Checkpoint backup SKIPPED (warm-start resume mode)")


# =============================================================================
# 7. Monkey-patch AdamW to force higher weight decay (1e-2)
# =============================================================================

_OrigAdamW = torch.optim.AdamW

class RegularizedAdamW(_OrigAdamW):
    """AdamW with forced higher weight decay for hybrid models."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, **kwargs):
        actual_wd = 1e-2  # v3: 1e-2 (was 5e-3 in v2)
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=actual_wd, amsgrad=amsgrad, **kwargs)
        print(f"  [v3] AdamW weight_decay forced to {actual_wd} (was {weight_decay})")

torch.optim.AdamW = RegularizedAdamW


# =============================================================================
# Run
# =============================================================================

print("=" * 60)
print("WIDER HYBRID v3 - GENERALIZATION OPTIMIZED")
print("  CNN backbone:    spatial=(32,64,128,256) temporal=256 -> 256d")
print("  Transformer:     d_model=256, 8 heads, 3 layers -> 256d")
print("  Feature MLP:     17 engineered features -> 64d")
print("  Fusion:          [576] -> 128 -> 1")
print("  CHANGES FROM v2:")
print("    Model size:    ~15.5M -> ~6-8M params (halved CNN)")
print("    Dropout:       0.4 -> 0.5")
print("    Weight decay:  5e-3 -> 1e-2")
print("    Learning rate: 1e-4 -> 5e-5")
print("    Grad clip:     0.5 -> 0.3")
print("    Subsample:     5x -> 8x")
print("    Window:        expanding (max 15d) -> sliding (max 30d)")
print("    Epochs:        6 -> 2 (warm start)")
print("    Batch size:    256 -> 512")
print("=" * 60)

try:
    sys.argv = [
        'run_wider_hybrid_wf_v3.py',
        '--model', 'hybrid',
        '--epochs', '2',
        '--batch-size', '512',
        '--subsample-train', '5',
        '--max-train-days', '30',
        '--lr', '5e-5',
        '--device', 'cuda',
        '--warm-start',
        '--output-dir', str(RESULTS_DIR),
    ]

    twf.main()

finally:
    # Restore original AdamW
    torch.optim.AdamW = _OrigAdamW

    # Rename checkpoint to "wider_hybrid_v3" variant
    new_ckpts = list(RESULTS_DIR.glob('checkpoint_hybrid_*.json'))
    for ckpt in new_ckpts:
        if ckpt.with_suffix('.json.standard_hybrid_backup') not in [b for b, _ in renamed]:
            wider_name = ckpt.name.replace('checkpoint_hybrid_', 'checkpoint_wider_hybrid_v3_')
            wider_path = ckpt.parent / wider_name
            ckpt.rename(wider_path)
            print(f"Saved wider hybrid v3 checkpoint: {wider_name}")

    new_preds = list(RESULTS_DIR.glob('oos_predictions_hybrid_*.npz'))
    for pred in new_preds:
        ts = pred.name.split('hybrid_')[1] if 'hybrid_' in pred.name else pred.name
        if 'wider' not in pred.name:
            wider_name = f"oos_predictions_wider_hybrid_v3_{ts}"
            wider_path = pred.parent / wider_name
            if not wider_path.exists():
                pred.rename(wider_path)
                print(f"Saved wider hybrid v3 predictions: {wider_name}")

    new_ckpt_preds = list(RESULTS_DIR.glob('ckpt_preds_hybrid_*.npz'))
    for pred in new_ckpt_preds:
        ts = pred.name.split('hybrid_')[1] if 'hybrid_' in pred.name else pred.name
        if 'wider' not in pred.name:
            wider_name = f"ckpt_preds_wider_hybrid_v3_{ts}"
            wider_path = pred.parent / wider_name
            if not wider_path.exists():
                pred.rename(wider_path)
                print(f"Saved wider hybrid v3 ckpt preds: {wider_name}")

    for backup, original in renamed:
        if backup.exists():
            backup.rename(original)
            print(f"Restored standard hybrid checkpoint: {original.name}")
