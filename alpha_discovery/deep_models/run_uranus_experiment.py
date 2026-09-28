"""
Uranus 5090 Experiment: Wider CNN + Features + Extended Context Window

Architecture: WiderHybridModel (CNN + Transformer + Feature MLP fusion)
  - CNN backbone: spatial=(64,128,256,512) temporal=512 -> 512d
  - Transformer: d_model=256, 8 heads, 3 layers -> 256d
  - Feature MLP: 17 engineered features -> 64d
  - Fusion: [512+256+64]=832 -> 256 -> 1
  - Window size: 50 bars (5x standard, ~5 seconds of context)
  - ~18-20M params

Training: 100 days (Jul 15 - Nov 30), expanding window, single pass.
Goal: proof-of-concept to see if more context + features improves IC.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'alpha_discovery' / 'deep_models'))
os.chdir(PROJECT_ROOT / 'alpha_discovery' / 'deep_models')

import torch
import torch.nn as nn

# Import the base modules
import train_walkforward as twf
from book_spatial_cnn import BookSpatialCNN

# Try to import event transformer (may not exist on all nodes)
try:
    from event_transformer import EventTransformer
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False
    print("WARNING: EventTransformer not available. Using CNN-only mode.")


# =============================================================================
# 1. Define the experiment model
# =============================================================================

class UranusExperimentModel(nn.Module):
    """
    Wider CNN + extended context + engineered features.

    If EventTransformer is available: full hybrid (CNN + Transformer + Features)
    If not: CNN + Features only (still tests wider context + features)
    """
    def __init__(self, dropout=0.3, window_size=50):
        super().__init__()
        self.window_size = window_size

        # Wider CNN backbone: (64,128,256,512), temporal=512
        self.book_encoder = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=(64, 128, 256, 512),
            temporal_channels=512,
            dropout=dropout,
            num_classes=512,  # output embedding dim
        )

        encoder_dim = 512

        if HAS_TRANSFORMER:
            # Transformer for event sequence
            self.event_encoder = EventTransformer(
                d_model=256,
                nhead=8,
                num_layers=3,
                dim_feedforward=1024,
                dropout=dropout,
                num_classes=256,
                use_cnn_stem=True,
            )
            encoder_dim += 256

        # Feature MLP: 17 engineered features -> 64d
        self.feature_mlp = nn.Sequential(
            nn.Linear(17, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        encoder_dim += 64

        # Fusion head
        self.fusion_norm = nn.LayerNorm(encoder_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [UranusExperiment] Total params: {total_params:,}")
        print(f"  [UranusExperiment] Window size: {window_size} bars")
        print(f"  [UranusExperiment] Has transformer: {HAS_TRANSFORMER}")
        print(f"  [UranusExperiment] Encoder dim: {encoder_dim}")

    def forward(self, book_windows, event_seqs=None, event_lengths=None, engineered_features=None):
        # CNN on book snapshots
        book_feats = self.book_encoder(book_windows)  # (B, 512)

        parts = [book_feats]

        # Transformer on events (if available)
        if HAS_TRANSFORMER and event_seqs is not None:
            event_feats = self.event_encoder(event_seqs, event_lengths)  # (B, 256)
            parts.append(event_feats)

        # Feature MLP
        if engineered_features is not None:
            feat_feats = self.feature_mlp(engineered_features)  # (B, 64)
        else:
            feat_feats = torch.zeros(book_feats.shape[0], 64,
                                     device=book_feats.device, dtype=book_feats.dtype)
        parts.append(feat_feats)

        fused = torch.cat(parts, dim=-1)
        fused = self.fusion_norm(fused)
        return self.fusion_head(fused)


# =============================================================================
# 2. Monkey-patch build_model
# =============================================================================

_original_build_model = twf.build_model

def patched_build_model(model_type, device, window_size=20, augment=False):
    """Build UranusExperimentModel with wider context."""
    if model_type == 'hybrid':
        model = UranusExperimentModel(dropout=0.3, window_size=50)
        model = model.to(device)
        return model
    else:
        return _original_build_model(model_type, device, window_size, augment)

twf.build_model = patched_build_model


# =============================================================================
# 3. Monkey-patch _forward_batch for features
# =============================================================================

def patched_forward_batch(model, batch, model_type, device):
    """Forward pass with engineered features."""
    if model_type == 'hybrid':
        if len(batch) == 5:
            book_windows, event_seqs, event_lengths, eng_features, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            eng_features = eng_features.to(device)
            targets = targets.to(device)
            preds = model(book_windows, event_seqs, event_lengths, eng_features).squeeze(-1)
        elif len(batch) == 4:
            book_windows, event_seqs, event_lengths, targets = batch
            book_windows = book_windows.to(device)
            event_seqs = event_seqs.to(device)
            event_lengths = event_lengths.to(device)
            targets = targets.to(device)
            preds = model(book_windows, event_seqs, event_lengths).squeeze(-1)
        else:
            # Book-only mode
            book_windows, targets = batch[0], batch[-1]
            book_windows = book_windows.to(device)
            targets = targets.to(device)
            preds = model(book_windows).squeeze(-1)
        return preds, targets
    else:
        # Delegate to original for non-hybrid
        return twf._original_forward_batch(model, batch, model_type, device)

# Save original before patching
if not hasattr(twf, '_original_forward_batch'):
    twf._original_forward_batch = twf._forward_batch
twf._forward_batch = patched_forward_batch


# =============================================================================
# 4. Set window_size=50 in dataset creation
# =============================================================================

_original_create_dataset = getattr(twf, 'create_dataset', None) or getattr(twf, '_create_dataset', None)

# The window_size is passed as an arg to train_walkforward via --window-size
# We'll add it to sys.argv below


# =============================================================================
# 5. Output directory
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'uranus_experiment'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("URANUS 5090 EXPERIMENT — Wider CNN + Features + Extended Context")
    print("  CNN backbone:     spatial=(64,128,256,512) temporal=512 -> 512d")
    print("  Transformer:      d_model=256, 8h, 3L -> 256d" if HAS_TRANSFORMER else "  Transformer: DISABLED")
    print("  Feature MLP:      17 features -> 128 -> 64d")
    print("  Fusion:           [832] -> 256 -> 64 -> 1")
    print("  Window size:      50 bars (~5 seconds of microstructure)")
    print("  Training window:  Jul 15 - Nov 30 (100 days)")
    print("  Epochs:           15")
    print("  Batch size:       256 (larger model, conservative batch)")
    print("  Dropout:          0.3")
    print("  Subsample:        3x")
    print("=" * 70)

    sys.argv = [
        'run_uranus_experiment.py',
        '--model', 'hybrid',
        '--epochs', '15',
        '--batch-size', '256',
        '--subsample-train', '3',
        '--min-train-days', '90',
        '--lr', '1e-4',
        '--device', 'cuda',
        '--output-dir', str(RESULTS_DIR),
    ]

    twf.main()
