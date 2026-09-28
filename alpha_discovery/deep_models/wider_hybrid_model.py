"""
Wider Hybrid Model — Wider CNN backbone + improved Transformer + engineered features.

Combines three streams via late fusion:
    1. Wider BookSpatialCNN (64,128,256,512) → 512-dim spatial embedding
    2. Transformer (d_model=256, 8 heads, 3 layers) → 256-dim event embedding
    3. Feature MLP (17 engineered features) → 64-dim feature embedding

Fusion: [512 + 256 + 64] = 832 → MLP → 1 (regression prediction)

Expected parameter count: ~15-18M.

This module is self-contained — it imports BookSpatialCNN and EventTransformer
from the existing codebase but does NOT modify them. The wider CNN backbone is
created by passing wider channel args to BookSpatialCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from book_spatial_cnn import BookSpatialCNN
from event_transformer import EventTransformer


class WiderHybridModel(nn.Module):
    """
    Wider CNN backbone (64,128,256,512) + Transformer with engineered features.

    Inputs:
        book_window: (B, T, 20, 4) — book snapshots (same as BookSpatialCNN)
        event_seq: (B, 200, 5) — raw event tokens (same as EventTransformer)
        event_features: (B, 17) — engineered features from event_features.py

    Architecture:
        1. Wider BookSpatialCNN processes book_window → (B, 512) spatial embedding
        2. Transformer processes event_seq → (B, 256) event embedding
        3. Feature MLP processes event_features → (B, 64) feature embedding
        4. Concat [512 + 256 + 64] = 832 → Fusion MLP → prediction

    Args:
        book_enc_dim: Output dim of CNN encoder (default: 512)
        event_enc_dim: Output dim of Transformer encoder (default: 256)
        engineered_dim: Number of engineered features (default: 17)
        eng_enc_dim: Projection dim for engineered features (default: 64)
        dropout: Dropout rate (default: 0.1)
        num_classes: Output classes (default: 1 for regression)
        window_size: Book window size (default: 20)
    """

    def __init__(
        self,
        book_enc_dim: int = 512,
        event_enc_dim: int = 256,
        engineered_dim: int = 17,
        eng_enc_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 1,
        window_size: int = 20,
    ):
        super().__init__()
        self.engineered_dim = engineered_dim

        # ---- 1. Wider CNN backbone ----
        # Standard BookSpatialCNN is (32,64,128,256) + temporal=256
        # Wider version: (64,128,256,512) + temporal=512
        self.book_encoder = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=(64, 128, 256, 512),
            temporal_channels=512,
            dropout=dropout,
            num_classes=book_enc_dim,  # output is the embedding dimension
        )

        # ---- 2. Improved Transformer ----
        # Wider & deeper than current hybrid: d_model=256, 8 heads, 3 layers
        self.event_encoder = EventTransformer(
            d_model=256,
            nhead=8,
            num_layers=3,
            dim_feedforward=1024,
            dropout=dropout,
            num_classes=event_enc_dim,  # output is the embedding dimension
            use_cnn_stem=True,
        )

        # ---- 3. Feature MLP ----
        self.feature_mlp = nn.Sequential(
            nn.Linear(engineered_dim, eng_enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(eng_enc_dim, eng_enc_dim),
        )

        # ---- 4. Fusion ----
        fusion_dim = book_enc_dim + event_enc_dim + eng_enc_dim  # 512 + 256 + 64 = 832
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        book_windows: torch.Tensor,                     # (B, window_size, 20, 4)
        event_seqs: torch.Tensor,                        # (B, seq_len, 5) int64
        event_lengths: torch.Tensor,                     # (B,)
        engineered_features: Optional[torch.Tensor] = None,  # (B, 17)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, num_classes)
        """
        # 1. CNN encodes book snapshots
        book_feats = self.book_encoder(book_windows)           # (B, 512)

        # 2. Transformer encodes event sequences
        event_feats = self.event_encoder(event_seqs, event_lengths)  # (B, 256)

        # 3. Feature MLP encodes engineered features
        if engineered_features is not None:
            feat_feats = self.feature_mlp(engineered_features)  # (B, 64)
        else:
            # Fallback: zeros if no features provided
            feat_feats = torch.zeros(
                book_feats.shape[0], 64,
                device=book_feats.device, dtype=book_feats.dtype,
            )

        # 4. Late fusion
        fused = torch.cat([book_feats, event_feats, feat_feats], dim=-1)  # (B, 832)
        fused = self.fusion_norm(fused)
        return self.fusion_head(fused)  # (B, num_classes)


if __name__ == '__main__':
    # Quick architecture verification
    model = WiderHybridModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"WiderHybridModel parameters: {n_params:,}")

    # Breakdown by component
    book_params = sum(p.numel() for p in model.book_encoder.parameters())
    event_params = sum(p.numel() for p in model.event_encoder.parameters())
    feat_params = sum(p.numel() for p in model.feature_mlp.parameters())
    fusion_params = sum(p.numel() for p in model.fusion_norm.parameters()) + \
                    sum(p.numel() for p in model.fusion_head.parameters())
    print(f"  Book encoder:   {book_params:,}")
    print(f"  Event encoder:  {event_params:,}")
    print(f"  Feature MLP:    {feat_params:,}")
    print(f"  Fusion head:    {fusion_params:,}")

    # Test forward pass
    B = 4
    book = torch.randn(B, 20, 20, 4)
    events = torch.randint(0, 8, (B, 200, 5))
    lengths = torch.randint(50, 200, (B,))
    eng_feats = torch.randn(B, 17)

    logits = model(book, events, lengths, eng_feats)
    print(f"\nInput shapes: book={book.shape}, events={events.shape}, features={eng_feats.shape}")
    print(f"Output shape: {logits.shape}")  # (4, 1)
