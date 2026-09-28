"""
Next-Gen CNN Architecture Experiments — Long-Context Models for Strategy Hold Times
====================================================================================
Designed 2026-03-29 for Razer (RTX 3070 8GB).

Problem:
  Current BookSpatialCNN window=20 (2s context) predicts 10s ahead.
  Strategies hold 10-91 MINUTES. Signal could flip 100 times before exit.
  We need:
    (a) LONGER context windows — capture structure that builds over minutes
    (b) LONGER prediction horizons — 30s, 60s, 120s, 300s
    (c) SMOOTHER signals — consistent across hold duration, not just 10s spike

Key constraints (Razer RTX 3070 8GB VRAM):
  - batch_size small (32-64 for large windows)
  - Models < 20M params preferred (transformer memory scales O(T^2))
  - float16/mixed precision mandatory for windows > 200 bars
  - num_workers=8, pin_memory=True always

Leakage rules (ABSOLUTE — Session 38 lesson):
  - Expanding window only (no sliding, no capped window)
  - CONCAT IC is the primary metric — per-fold IC is inflated for temporal models
  - All normalization stats from training split only
  - No cross-day sequences

Architectures:
  A. WiderCNN-LongWindow  — baseline bigger window, same arch (w=50,100,200)
  B. CausalAttentionCNN   — spatial CNN per frame → causal MHA (already coded)
  C. HierarchicalCNN      — short CNN (w=20) → second CNN/LSTM on CNN outputs
  D. MultiScaleCNN        — parallel streams at 2s/10s/30s timescales
  E. LightTransformer     — tiny transformer directly on MBO features (no CNN)

Prediction horizons (matching strategy hold times):
  h1 = 100 bars = 10s   (current baseline)
  h2 = 300 bars = 30s
  h3 = 600 bars = 60s
  h4 = 1200 bars = 120s
  h5 = 3000 bars = 300s (5 min)
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from book_spatial_cnn import BookSpatialCNN, SpatialResBlock, TemporalResBlock

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BOOK_CACHE_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache')

N_FOLDS        = 10    # quick diagnostic — 10 folds only
MIN_TRAIN_DAYS = 10
PURGE_DAYS     = 1
TICK_SIZE      = 0.25
NUM_WORKERS    = 8
PIN_MEMORY     = True

# Horizon bars → seconds mapping (at 100ms bars)
HORIZON_MAP = {
    '10s':   100,
    '30s':   300,
    '60s':   600,
    '120s':  1200,
    '300s':  3000,
}

# ─────────────────────────────────────────────────────────────────────────────
# Architecture A: Wider CNN with larger window (baseline extension)
# ─────────────────────────────────────────────────────────────────────────────

class LongWindowCNN(nn.Module):
    """
    BookSpatialCNN with larger temporal window but SMALLER spatial channels
    to keep VRAM in budget for Razer 8GB.

    Unlike the 'wider' CNN (which widened channels), this version keeps spatial
    channels same as baseline but extends the temporal window and adds an extra
    temporal ResBlock to capture longer patterns.

    w=50  → 5s context,  ~5M params
    w=100 → 10s context, ~5M params (same — temporal layers scale linearly)
    w=200 → 20s context, ~5M params
    w=500 → 50s context, ~5M params

    Key change vs baseline: temporal stem uses dilated convolutions at larger
    windows to capture multi-scale temporal structure without quadratic cost.
    """
    def __init__(
        self,
        window_size: int = 100,
        spatial_channels: Tuple[int, ...] = (32, 64, 128, 256),
        temporal_channels: int = 256,
        dropout: float = 0.15,
        num_classes: int = 1,
        use_dilated: bool = True,
    ):
        super().__init__()
        self.window_size = window_size

        # ---- Spatial encoder (same as baseline, leaner channels) ----
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(1, spatial_channels[0], (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_channels[0]),
            nn.GELU(),
        )
        spatial_layers = []
        for i in range(len(spatial_channels) - 1):
            spatial_layers.append(SpatialResBlock(spatial_channels[i], spatial_channels[i+1], dropout * 0.5))
        self.spatial_blocks = nn.Sequential(*spatial_layers)
        self.spatial_pool   = nn.AdaptiveAvgPool2d((20, 1))
        spatial_out         = spatial_channels[-1] * 20

        self.spatial_compress = nn.Sequential(
            nn.Linear(spatial_out, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Bid/ask side convolutions ----
        self.bid_conv = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.ask_conv = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1))

        temporal_in = 320  # 256 + 64

        # ---- Temporal processor with DILATED convolutions for long windows ----
        if use_dilated and window_size > 50:
            # Dilated temporal convolutions capture long-range dependencies efficiently
            # Dilation 1,2,4 gives effective receptive field of ~24 bars
            self.temporal_stem = nn.Sequential(
                nn.Conv1d(temporal_in, temporal_channels, 5, padding=2, bias=False),
                nn.BatchNorm1d(temporal_channels),
                nn.GELU(),
            )
            self.temporal_blocks = nn.ModuleList([
                # Standard
                TemporalResBlock(temporal_channels, kernel_size=5, dropout=dropout),
                # Dilated × 2
                _DilatedTemporalBlock(temporal_channels, kernel_size=5, dilation=2, dropout=dropout),
                # Dilated × 4
                _DilatedTemporalBlock(temporal_channels, kernel_size=5, dilation=4, dropout=dropout),
                # Dilated × 8 for very long windows
                _DilatedTemporalBlock(temporal_channels, kernel_size=5, dilation=8, dropout=dropout),
            ])
        else:
            self.temporal_stem = nn.Sequential(
                nn.Conv1d(temporal_in, temporal_channels, 5, padding=2, bias=False),
                nn.BatchNorm1d(temporal_channels),
                nn.GELU(),
            )
            self.temporal_blocks = nn.ModuleList([
                TemporalResBlock(temporal_channels, kernel_size=5, dropout=dropout),
                TemporalResBlock(temporal_channels, kernel_size=3, dropout=dropout),
            ])

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Head ----
        self.head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"LongWindowCNN: window={window_size}, dilated={use_dilated}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        # Spatial encoding per frame
        xs = x.reshape(B * T, 1, L, F)
        xs = self.spatial_stem(xs)
        xs = self.spatial_blocks(xs)
        xs = self.spatial_pool(xs)
        xs = xs.reshape(B * T, -1)
        xs = self.spatial_compress(xs)
        xs = xs.reshape(B, T, -1)

        # Bid/ask features
        bid = x[:, :, :10, :].reshape(B * T, 10, F).permute(0, 2, 1)
        ask = x[:, :, 10:, :].reshape(B * T, 10, F).permute(0, 2, 1)
        bf  = self.bid_conv(bid).squeeze(-1).reshape(B, T, -1)
        af  = self.ask_conv(ask).squeeze(-1).reshape(B, T, -1)

        combined = torch.cat([xs, bf, af], dim=-1).permute(0, 2, 1)  # (B, 320, T)

        out = self.temporal_stem(combined)
        for block in self.temporal_blocks:
            out = block(out)
        out = self.temporal_pool(out).squeeze(-1)
        return self.head(out).squeeze(-1)


class _DilatedTemporalBlock(nn.Module):
    """Dilated 1D residual block for long-range temporal modeling."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 2, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture C: Hierarchical CNN — CNN-on-CNN outputs
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalCNN(nn.Module):
    """
    Two-stage hierarchical architecture:

    Stage 1: Short-window spatial CNN (w=20, 2s) → 64-dim embedding per 2s chunk
    Stage 2: Second CNN over the sequence of stage-1 embeddings
             With T_long=300 and chunk=20, we have 15 stage-1 embeddings covering 30s

    This allows Stage 2 to learn patterns across 30-90s using Stage 1's compressed
    representations of each 2s window.

    Memory efficient: Stage 1 processes 20 bars at a time — no 300-bar windows.
    ~8M params, fits 8GB VRAM.
    """
    def __init__(
        self,
        chunk_size:   int   = 20,    # Stage 1: 2-second chunks
        n_chunks:     int   = 15,    # Stage 2: 15 chunks = 30s total
        embed_dim:    int   = 64,    # Stage 1 output embedding dim
        temporal_dim: int   = 128,   # Stage 2 channels
        dropout:      float = 0.15,
        num_classes:  int   = 1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_chunks   = n_chunks
        self.window_size = chunk_size * n_chunks  # 300 total bars

        # ---- Stage 1: Spatial CNN per chunk (same as baseline but compressed output) ----
        self.stage1_spatial_stem = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1_spatial = nn.Sequential(
            SpatialResBlock(32, 64, dropout * 0.5),
            SpatialResBlock(64, 128, dropout * 0.5),
        )
        self.stage1_pool    = nn.AdaptiveAvgPool2d((20, 1))

        self.stage1_compress = nn.Sequential(
            nn.Linear(128 * 20, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage1_temporal = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            TemporalResBlock(256, kernel_size=3, dropout=dropout),
        )
        self.stage1_pool_t = nn.AdaptiveAvgPool1d(1)
        # Project to embed_dim for stage 2
        self.stage1_proj = nn.Linear(256, embed_dim)

        # ---- Stage 2: Temporal CNN over stage-1 embeddings ----
        self.stage2_stem = nn.Sequential(
            nn.Conv1d(embed_dim, temporal_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),
        )
        self.stage2_blocks = nn.ModuleList([
            TemporalResBlock(temporal_dim, kernel_size=3, dropout=dropout),
            _DilatedTemporalBlock(temporal_dim, kernel_size=3, dilation=2, dropout=dropout),
            _DilatedTemporalBlock(temporal_dim, kernel_size=3, dilation=4, dropout=dropout),
        ])
        self.stage2_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Head ----
        self.head = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"HierarchicalCNN: chunk={chunk_size}, n_chunks={n_chunks}, "
              f"total_window={self.window_size}, params={n_params:,}")

    def _encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode a (B, chunk_size, 20, 4) chunk → (B, embed_dim)."""
        B, T, L, F = x_chunk.shape
        # Spatial
        xs = x_chunk.reshape(B * T, 1, L, F)
        xs = self.stage1_spatial_stem(xs)
        xs = self.stage1_spatial(xs)
        xs = self.stage1_pool(xs)
        xs = xs.reshape(B * T, -1)
        xs = self.stage1_compress(xs)       # (B*T, 256)
        xs = xs.reshape(B, T, -1)          # (B, T, 256)

        # Short temporal processing within chunk
        xt = xs.permute(0, 2, 1)            # (B, 256, T)
        xt = self.stage1_temporal(xt)       # (B, 256, T)
        xt = self.stage1_pool_t(xt).squeeze(-1)  # (B, 256)
        return self.stage1_proj(xt)          # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        assert T == self.window_size, f"Expected {self.window_size} bars, got {T}"

        # Encode each chunk
        chunk_embeddings = []
        for c in range(self.n_chunks):
            chunk = x[:, c * self.chunk_size : (c + 1) * self.chunk_size, :, :]
            emb   = self._encode_chunk(chunk)   # (B, embed_dim)
            chunk_embeddings.append(emb)

        # Stack: (B, n_chunks, embed_dim) → (B, embed_dim, n_chunks)
        seq = torch.stack(chunk_embeddings, dim=1).permute(0, 2, 1)

        # Stage 2 temporal processing
        out = self.stage2_stem(seq)
        for block in self.stage2_blocks:
            out = block(out)
        out = self.stage2_pool(out).squeeze(-1)

        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture D: Multi-Scale CNN — parallel streams at 2s/10s/30s
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    """
    Three parallel CNN streams with different temporal resolutions:
      Stream A: window=20  (2s)   — fine-grained microstructure
      Stream B: window=100 (10s)  — medium-term momentum
      Stream C: window=300 (30s)  — macro context

    Each stream produces a 128-dim feature vector.
    Features concatenated → fusion MLP → prediction.

    Total ~12M params, fits 8GB VRAM.
    Key advantage: explicitly models the multi-scale nature of order flow.
    """
    def __init__(
        self,
        num_classes:  int   = 1,
        dropout:      float = 0.15,
    ):
        super().__init__()

        self.window_size = 300  # must load 300-bar windows; streams use subsets

        # Shared spatial encoder (applied to each bar regardless of stream)
        # Smaller to save VRAM: (16, 32, 64, 128) instead of (32, 64, 128, 256)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.spatial_blocks = nn.Sequential(
            SpatialResBlock(16, 32, dropout * 0.5),
            SpatialResBlock(32, 64, dropout * 0.5),
            SpatialResBlock(64, 128, dropout * 0.5),
        )
        self.spatial_pool     = nn.AdaptiveAvgPool2d((20, 1))
        self.spatial_compress = nn.Sequential(nn.Linear(128 * 20, 128), nn.GELU(), nn.Dropout(dropout * 0.5))
        # Output per frame: 128-dim

        # Stream A: last 20 bars (2s) — fine-grained
        self.stream_a = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            TemporalResBlock(128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        # Stream B: last 100 bars (10s) — dilated for range
        self.stream_b = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            TemporalResBlock(128, kernel_size=5, dropout=dropout),
            _DilatedTemporalBlock(128, kernel_size=5, dilation=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        # Stream C: all 300 bars (30s) — heavily dilated
        self.stream_c = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            TemporalResBlock(128, kernel_size=5, dropout=dropout),
            _DilatedTemporalBlock(128, kernel_size=5, dilation=2, dropout=dropout),
            _DilatedTemporalBlock(128, kernel_size=5, dilation=4, dropout=dropout),
            _DilatedTemporalBlock(128, kernel_size=5, dilation=8, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        # Fusion: 128*3 = 384 → head
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MultiScaleCNN: window=300 (streams: 2s/10s/30s), params={n_params:,}")

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all frames spatially: (B, T, 20, 4) → (B, T, 128)."""
        B, T, L, F = x.shape
        xs = x.reshape(B * T, 1, L, F)
        xs = self.spatial_stem(xs)
        xs = self.spatial_blocks(xs)
        xs = self.spatial_pool(xs)
        xs = xs.reshape(B * T, -1)
        xs = self.spatial_compress(xs)
        return xs.reshape(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        assert T == 300, f"Expected 300 bars for multi-scale, got {T}"

        # Encode all bars (shared spatial encoder)
        all_feats = self._encode_frames(x)  # (B, 300, 128)
        seq = all_feats.permute(0, 2, 1)     # (B, 128, 300)

        # Stream A: last 20 bars
        fa = self.stream_a(seq[:, :, -20:]).squeeze(-1)   # (B, 128)
        # Stream B: last 100 bars
        fb = self.stream_b(seq[:, :, -100:]).squeeze(-1)  # (B, 128)
        # Stream C: all 300 bars
        fc = self.stream_c(seq).squeeze(-1)               # (B, 128)

        fused = torch.cat([fa, fb, fc], dim=-1)           # (B, 384)
        return self.head(fused).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture E: Light Transformer directly on MBO-level book features
# (No CNN, pure attention over book state sequence)
# ─────────────────────────────────────────────────────────────────────────────

class LightBookTransformer(nn.Module):
    """
    Lightweight Causal Transformer directly on book features — no CNN.

    Input: (B, T, 20, 4) raw book → flatten to (B, T, 80) → linear projection
    Transformer: 4 layers, d_model=128, 4 heads — tiny but fast
    Max window: 1000 bars (100s) feasible on 8GB VRAM at batch=32

    Motivation: CNN was designed for spatial structure in a single snapshot.
    At longer timescales, the TEMPORAL patterns between snapshots matter more.
    Pure attention might find patterns the CNN's spatial inductive bias misses.

    ~2M params — fastest to train, quickest to diagnose.
    """
    def __init__(
        self,
        window_size:    int   = 500,
        d_model:        int   = 128,
        n_heads:        int   = 4,
        n_layers:       int   = 4,
        ffn_dim:        int   = 256,
        dropout:        float = 0.15,
        num_classes:    int   = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model     = d_model

        book_flat_dim = 20 * 4  # 80

        # Input projection: 80 raw book features → d_model
        self.input_proj = nn.Sequential(
            nn.LayerNorm(book_flat_dim),
            nn.Linear(book_flat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding (learned — more flexible than sinusoidal for long windows)
        self.pos_emb = nn.Embedding(window_size + 2, d_model)

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Head: use last frame
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"LightBookTransformer: window={window_size}, d_model={d_model}, "
              f"n_layers={n_layers}, n_heads={n_heads}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        assert T == self.window_size

        # Flatten book: (B, T, 20, 4) → (B, T, 80)
        x_flat = x.reshape(B, T, -1).float()
        # Normalize depth/orders/age (cols correspond to level features)
        # Simple approach: log1p the positive features
        # Feature layout: 4 features per level × 20 levels
        # cols [1,5,9,...] are depth (every 4th); [2,6,...] orders; [3,7,...] age
        # Easier: just layer-norm the whole thing
        emb = self.input_proj(x_flat)  # (B, T, d_model)

        # Positional encoding
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        emb = emb + self.pos_emb(pos)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        out = self.transformer(emb, mask=causal_mask)  # (B, T, d_model)
        last = out[:, -1, :]                            # (B, d_model)
        return self.head(last).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture F: Slim CausalTransformerCNN (Razer-safe version of Track 3)
# Same as CausalTransformerCNN but smaller (128-dim, 3 layers) for 8GB VRAM
# ─────────────────────────────────────────────────────────────────────────────

class SlimCausalTransformerCNN(nn.Module):
    """
    Reduced CausalTransformerCNN for Razer (RTX 3070 8GB).

    Same concept as temporal_cnn_300f_models.CausalTransformerCNN but:
    - transformer_dim = 128 (vs 256)
    - n_layers = 3 (vs 6)
    - n_heads = 4 (vs 8)
    - CNN spatial channels: (32, 64, 128, 256) leaner backbone

    ~8M params, safe for 300-frame windows at batch=32 on 8GB.
    """
    def __init__(
        self,
        window_size:      int   = 300,
        cnn_embed_dim:    int   = 256,
        transformer_dim:  int   = 128,
        n_heads:          int   = 4,
        n_layers:         int   = 3,
        ffn_dim:          int   = 256,
        dropout:          float = 0.15,
        num_classes:      int   = 1,
    ):
        super().__init__()
        self.window_size = window_size

        # Leaner spatial CNN per frame
        self.spatial_cnn = BookSpatialCNN(
            window_size=1,
            spatial_channels=(32, 64, 128, 256),
            temporal_channels=cnn_embed_dim,
            dropout=dropout,
            num_classes=cnn_embed_dim,
        )

        self.input_proj = nn.Sequential(
            nn.Linear(cnn_embed_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Learned positional embedding (more flexible for varied window sizes)
        self.pos_emb = nn.Embedding(window_size + 2, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SlimCausalTransformerCNN: window={window_size}, transformer_dim={transformer_dim}, "
              f"n_layers={n_layers}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        assert T == self.window_size

        x_flat   = x.reshape(B * T, 1, L, F)
        emb_flat = self.spatial_cnn(x_flat)         # (B*T, cnn_embed_dim)
        emb      = emb_flat.reshape(B, T, -1)        # (B, T, cnn_embed_dim)
        emb      = self.input_proj(emb)              # (B, T, transformer_dim)

        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        emb  = emb + self.pos_emb(pos)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out  = self.transformer(emb, mask=mask)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: multi-horizon support
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe_net(mid_prices, day_boundaries, horizon_bars=100, tick_size=0.25):
    from numpy.lib.stride_tricks import sliding_window_view
    N  = len(mid_prices)
    H  = horizon_bars
    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)
    valid_len = N - H - 1
    if valid_len > 0:
        windows = sliding_window_view(mid_prices[1:], H)[:valid_len]
        mfe_long[:valid_len]  = np.maximum(0.0, (windows.max(1) - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - windows.min(1)) / tick_size)
    for d in range(len(day_boundaries) - 2):
        day_end = day_boundaries[d + 1]
        nan_s   = max(day_boundaries[d], day_end - H)
        mfe_long[nan_s:day_end]  = np.nan
        mfe_short[nan_s:day_end] = np.nan
    return (mfe_long - mfe_short).astype(np.float32)


class LongWindowDataset(Dataset):
    """
    Dataset for long-window experiments.
    Supports any window_size and horizon_bars.
    float16 storage to manage RAM on Razer (16GB).
    """
    def __init__(self, day_data_list, target, day_boundaries,
                 window_size=300, subsample=1, horizon_bars=100):
        self.window_size = window_size
        all_t = []
        for d in day_data_list:
            t = d['book_tensors'].astype(np.float32)
            t[:, :, 1] = np.log1p(t[:, :, 1])
            t[:, :, 2] = np.log1p(t[:, :, 2])
            t[:, :, 3] = np.log1p(t[:, :, 3])
            all_t.append(t.astype(np.float16))
        self.tensors = np.concatenate(all_t, axis=0)
        self.target  = target

        valid = []
        for di in range(len(day_boundaries) - 1):
            s = day_boundaries[di]
            e = day_boundaries[di + 1]
            for i in range(s + window_size - 1, e - horizon_bars):
                if np.isfinite(target[i]):
                    valid.append(i)
        if subsample > 1:
            valid = valid[::subsample]
        self.valid = valid

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        w = self.tensors[i - self.window_size + 1 : i + 1].astype(np.float32)
        t = float(self.target[i])
        return torch.from_numpy(w), torch.tensor(t, dtype=torch.float32)


def load_book_days(cache_dir, dates):
    cache = Path(cache_dir)
    data_list, boundaries, all_mids = [], [0], []
    for date in dates:
        f = cache / f'{date}_book_tensors.npz'
        if not f.exists():
            continue
        npz = np.load(f)
        data_list.append(dict(npz))
        all_mids.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))
    mid_concat = np.concatenate(all_mids) if all_mids else np.array([])
    return data_list, mid_concat, boundaries


def get_dates(cache_dir):
    return sorted([f.name.replace('_book_tensors.npz', '')
                   for f in Path(cache_dir).glob('*_book_tensors.npz')])


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(model, train_loader, val_loader, device, n_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=n_epochs,
    )
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            if scaler:
                with torch.amp.autocast('cuda'):
                    p = model(xb).cpu().float().numpy()
            else:
                p = model(xb).cpu().numpy()
            preds_all.append(p)
            tgts_all.append(yb.numpy())

    preds   = np.concatenate(preds_all)
    targets = np.concatenate(tgts_all)
    mask    = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 10:
        return 0.0, preds, targets
    ic, _ = spearmanr(preds[mask], targets[mask])
    return float(ic) if np.isfinite(ic) else 0.0, preds, targets


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    # --- Quick ablation: standard CNN with more context, same 10s horizon ---
    'ABLATION_w200_h10':    dict(arch='longcnn',  window=200,  horizon='10s',  epochs=2, batch=64,  subsample=10),
    'ABLATION_w300_h10':    dict(arch='longcnn',  window=300,  horizon='10s',  epochs=2, batch=32,  subsample=10),

    # --- Architecture A: Wider/longer window CNN baseline ---
    'A1_longcnn_w50_h10':   dict(arch='longcnn',  window=50,   horizon='10s',  epochs=2, batch=128, subsample=10),
    'A2_longcnn_w100_h30':  dict(arch='longcnn',  window=100,  horizon='30s',  epochs=2, batch=64,  subsample=15),
    'A3_longcnn_w200_h60':  dict(arch='longcnn',  window=200,  horizon='60s',  epochs=2, batch=32,  subsample=20),
    'A4_longcnn_w500_h120': dict(arch='longcnn',  window=500,  horizon='120s', epochs=2, batch=16,  subsample=30),

    # --- Architecture B: Slim Causal Transformer + CNN ---
    'B1_slimtfm_w100_h30':  dict(arch='slim_tfm', window=100,  horizon='30s',  epochs=2, batch=64,  subsample=15),
    'B2_slimtfm_w300_h60':  dict(arch='slim_tfm', window=300,  horizon='60s',  epochs=2, batch=32,  subsample=20),
    'B2_slimtfm_w300_h30':  dict(arch='slim_tfm', window=300,  horizon='30s',  epochs=2, batch=32,  subsample=20),
    'B3_slimtfm_w300_h120': dict(arch='slim_tfm', window=300,  horizon='120s', epochs=2, batch=32,  subsample=20),

    # --- Architecture C: Hierarchical CNN ---
    'C1_hierarch_w300_h30':  dict(arch='hierarchical', window=300, horizon='30s',  epochs=2, batch=64,  subsample=15,
                                  extra=dict(chunk_size=20, n_chunks=15)),
    'C2_hierarch_w600_h120': dict(arch='hierarchical', window=600, horizon='120s', epochs=2, batch=32,  subsample=25,
                                  extra=dict(chunk_size=20, n_chunks=30)),

    # --- Architecture D: Multi-scale CNN ---
    'D1_multiscale_h30':  dict(arch='multiscale', window=300, horizon='30s',  epochs=2, batch=32, subsample=20),
    'D2_multiscale_h120': dict(arch='multiscale', window=300, horizon='120s', epochs=2, batch=32, subsample=20),

    # --- Architecture E: Light Transformer on raw book features ---
    'E1_light_tfm_w200_h30':  dict(arch='light_tfm', window=200,  horizon='30s',  epochs=2, batch=64,  subsample=15),
    'E2_light_tfm_w500_h60':  dict(arch='light_tfm', window=500,  horizon='60s',  epochs=2, batch=32,  subsample=20),
    'E3_light_tfm_w1000_h120':dict(arch='light_tfm', window=1000, horizon='120s', epochs=2, batch=16,  subsample=30),
}


def build_model(exp_cfg: dict) -> nn.Module:
    arch = exp_cfg['arch']
    window = exp_cfg['window']
    extra = exp_cfg.get('extra', {})

    if arch == 'longcnn':
        return LongWindowCNN(window_size=window, use_dilated=(window > 50))
    elif arch == 'slim_tfm':
        return SlimCausalTransformerCNN(window_size=window)
    elif arch == 'hierarchical':
        chunk  = extra.get('chunk_size', 20)
        n_ch   = extra.get('n_chunks', window // 20)
        return HierarchicalCNN(chunk_size=chunk, n_chunks=n_ch)
    elif arch == 'multiscale':
        return MultiScaleCNN()
    elif arch == 'light_tfm':
        return LightBookTransformer(window_size=window)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_name: str, n_folds: int = N_FOLDS):
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS)}")

    cfg       = EXPERIMENTS[exp_name]
    window    = cfg['window']
    horizon_s = cfg['horizon']
    horizon   = HORIZON_MAP[horizon_s]
    epochs    = cfg['epochs']
    batch     = cfg['batch']
    subsample = cfg['subsample']

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = SCRIPT_DIR / 'results' / f'nextgen_{exp_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"NEXT-GEN EXPERIMENT: {exp_name}")
    print(f"  Arch:     {cfg['arch']}")
    print(f"  Window:   {window} bars = {window/10:.0f}s context")
    print(f"  Horizon:  {horizon} bars = {horizon_s}")
    print(f"  Folds:    {n_folds}, Epochs: {epochs}, Batch: {batch}, Subsample: {subsample}")
    print(f"  Device:   {device}")
    print(f"{'='*65}\n")

    # MLflow
    mlrun = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('NextGen_CNN')
            mlrun = mlflow.start_run(run_name=f'{exp_name}_{timestamp}')
            mlflow.log_params({
                'exp_name': exp_name,
                'arch': cfg['arch'],
                'window_size': window,
                'horizon_bars': horizon,
                'horizon_str': horizon_s,
                'n_folds': n_folds,
                'epochs': epochs,
                'batch_size': batch,
                'subsample': subsample,
            })
        except Exception as e:
            print(f"MLflow init failed (non-fatal): {e}")
            mlrun = None

    dates = get_dates(BOOK_CACHE_DIR)
    print(f"Available dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    needed = n_folds + MIN_TRAIN_DAYS + PURGE_DAYS
    if len(dates) < needed:
        raise ValueError(f"Need {needed} dates, have {len(dates)}")
    use_dates = dates[-needed:]

    fold_ics, all_preds, all_tgts = [], [], []

    for fi in range(n_folds):
        test_idx    = MIN_TRAIN_DAYS + PURGE_DAYS + fi
        # EXPANDING WINDOW: train on ALL dates up to test - 1
        train_dates = use_dates[:MIN_TRAIN_DAYS + fi]
        test_dates  = [use_dates[test_idx]]

        print(f"\n--- Fold {fi+1}/{n_folds} | Train: {train_dates[0]}..{train_dates[-1]} "
              f"({len(train_dates)}d) | Test: {test_dates[0]}")
        t0 = time.time()

        tr_data, tr_mids, tr_bounds = load_book_days(BOOK_CACHE_DIR, train_dates)
        if not tr_data:
            print("  No train data, skipping")
            continue

        tr_target = compute_mfe_net(tr_mids, tr_bounds, horizon, TICK_SIZE)
        fin_mask  = np.isfinite(tr_target)
        if fin_mask.sum() < 200:
            print(f"  Too few samples ({fin_mask.sum()}), skipping")
            continue

        tgt_mean  = float(tr_target[fin_mask].mean())
        tgt_std   = float(tr_target[fin_mask].std()) or 1.0
        tr_target = (tr_target - tgt_mean) / tgt_std

        train_ds = LongWindowDataset(tr_data, tr_target, tr_bounds,
                                     window_size=window, subsample=subsample,
                                     horizon_bars=horizon)
        del tr_data, tr_mids, tr_target, tr_bounds
        gc.collect()

        te_data, te_mids, te_bounds = load_book_days(BOOK_CACHE_DIR, test_dates)
        if not te_data:
            continue

        te_target = compute_mfe_net(te_mids, te_bounds, horizon, TICK_SIZE)
        te_target = (te_target - tgt_mean) / tgt_std
        test_ds   = LongWindowDataset(te_data, te_target, te_bounds,
                                      window_size=window, subsample=1,
                                      horizon_bars=horizon)
        # Keep te_mids/te_bounds for multi-horizon IC eval
        _te_mids, _te_bounds = te_mids, te_bounds
        del te_data
        gc.collect()

        if len(train_ds) < 50 or len(test_ds) < 5:
            print(f"  Dataset too small (train={len(train_ds)}, test={len(test_ds)}), skipping")
            continue

        print(f"  Train samples: {len(train_ds):,} | Test samples: {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model = build_model(cfg).to(device)
        ic, preds, tgts = train_fold(model, train_loader, test_loader, device, epochs)
        fold_ics.append(ic)
        all_preds.append(preds)
        all_tgts.append(tgts)

        elapsed = time.time() - t0

        # Multi-horizon IC: measure prediction quality at ALL horizons
        EVAL_HORIZONS = {'10s': 100, '30s': 300, '60s': 600, '120s': 1200}
        horizon_ics = {}
        # Get the valid indices used by test_ds to align predictions with targets
        test_valid_idx = test_ds.valid
        for h_name, h_bars in EVAL_HORIZONS.items():
            alt_target = compute_mfe_net(_te_mids, _te_bounds, h_bars, TICK_SIZE)
            # Extract targets at same indices as predictions
            alt_vals = alt_target[test_valid_idx]
            mask_h = np.isfinite(preds) & np.isfinite(alt_vals)
            if mask_h.sum() >= 10:
                h_ic, _ = spearmanr(preds[mask_h], alt_vals[mask_h])
                horizon_ics[h_name] = float(h_ic) if np.isfinite(h_ic) else 0.0
            else:
                horizon_ics[h_name] = 0.0
        del _te_mids, _te_bounds
        gc.collect()

        train_h = horizon_s
        ic_line = " | ".join(f"{h}={horizon_ics.get(h,0):.4f}{'*' if h==train_h else ''}"
                             for h in ['10s','30s','60s','120s'])
        print(f"  Fold {fi+1} IC={ic:.4f} | {elapsed:.0f}s")
        print(f"  Multi-horizon IC: {ic_line}  (* = training horizon)")

        # Save artifacts (both .pt and .npz per leakage checklist)
        torch.save(model.state_dict(),
                   str(out_dir / f'{exp_name}_fold_{fi+1:03d}_{timestamp}.pt'))
        np.savez(str(out_dir / f'{exp_name}_fold_{fi+1:03d}_{timestamp}_preds.npz'),
                 predictions=preds, targets=tgts,
                 fold=fi+1, ic=ic, test_date=test_dates[0],
                 exp_name=exp_name, arch=cfg['arch'],
                 window_size=window, horizon_bars=horizon,
                 horizon_ics=json.dumps(horizon_ics))

        if MLFLOW_AVAILABLE and mlrun:
            try:
                metrics = {'fold_ic': ic, 'elapsed_s': elapsed}
                for h_name, h_ic in horizon_ics.items():
                    metrics[f'ic_{h_name}'] = h_ic
                mlflow.log_metrics(metrics, step=fi+1)
            except Exception:
                pass

        del model, train_ds, test_ds, train_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    if fold_ics:
        agg_ic    = float(np.mean(fold_ics))
        ic_std    = float(np.std(fold_ics))
        icir      = agg_ic / ic_std if ic_std > 1e-8 else 0.0
        all_p_cat = np.concatenate(all_preds)
        all_t_cat = np.concatenate(all_tgts)
        mask      = np.isfinite(all_p_cat) & np.isfinite(all_t_cat)
        concat_ic = float(spearmanr(all_p_cat[mask], all_t_cat[mask])[0])
        baseline  = 0.145  # WiderCNN w=20 baseline

        print(f"\n{'='*65}")
        print(f"RESULTS: {exp_name}")
        print(f"  Arch:         {cfg['arch']}, window={window} ({window/10:.0f}s), horizon={horizon_s}")
        print(f"  Folds:        {len(fold_ics)}/{n_folds}")
        print(f"  Mean fold IC: {agg_ic:.4f} ± {ic_std:.4f}")
        print(f"  IC-IR:        {icir:.4f}")
        print(f"  Concat IC:    {concat_ic:.4f}  ← PRIMARY METRIC (inflation-free)")
        print(f"  Baseline:     {baseline:.3f} (WiderCNN w=20, horizon=10s)")
        print(f"  Lift vs baseline: {concat_ic - baseline:+.4f}")
        print(f"  Leakage audit: PENDING — must verify expanding window used correctly")
        print(f"{'='*65}")

        summary = {
            'experiment':    exp_name,
            'arch':          cfg['arch'],
            'window_size':   window,
            'window_seconds': window / 10.0,
            'horizon_bars':  horizon,
            'horizon_str':   horizon_s,
            'n_folds':       len(fold_ics),
            'fold_ics':      fold_ics,
            'mean_fold_ic':  agg_ic,
            'ic_std':        ic_std,
            'icir':          icir,
            'concat_ic':     concat_ic,
            'baseline_ic':   baseline,
            'lift_vs_baseline': concat_ic - baseline,
            'timestamp':     timestamp,
            'leakage_audit': 'PENDING',
        }
        sp = out_dir / f'summary_{timestamp}.json'
        with open(sp, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {sp}")

        if MLFLOW_AVAILABLE and mlrun:
            try:
                mlflow.log_metrics({
                    'agg_ic': agg_ic, 'ic_std': ic_std, 'icir': icir,
                    'concat_ic': concat_ic, 'lift': concat_ic - baseline,
                })
                mlflow.log_artifact(str(sp))
                mlflow.end_run()
            except Exception:
                pass

    return fold_ics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Next-Gen CNN Architecture Experiments')
    parser.add_argument('--exp', required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment to run')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help=f'Number of folds (default {N_FOLDS})')
    args = parser.parse_args()

    print(f"\nRunning experiment: {args.exp}")
    print(f"Available experiments: {list(EXPERIMENTS.keys())}")
    run_experiment(args.exp, n_folds=args.n_folds)
