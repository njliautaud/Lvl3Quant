"""
TRACK 3: Temporal CNN Models — 30 Seconds of Context (300 frames)
=================================================================
Three architectures that read 300 consecutive 100ms book snapshots (30s of context)
and predict 10 seconds ahead (mfe_net at 100 bars).

Design goal: detect patterns that BUILD over 30 seconds — not just the instantaneous
book state. The user hypothesis: conviction accumulates slowly and shows in the book
structure before a move. A window=20 (2s) model misses this.

Architecture 1: CNN + Causal Attention (CausalAttentionCNN)
  - Proven WiderBookSpatialCNN extracts 512-dim spatial embedding per frame
  - 300 frame embeddings → Causal Multi-Head Attention (8 heads)
  - Positional encoding (sinusoidal) so model knows temporal distance
  - Causal mask: each frame attends only to itself and prior frames
  - ~14M params

Architecture 2: 3D CNN (ThreeDimensionalCNN)
  - Input: (B, 300, 20, 4) book window
  - 3D convolutions: kernels span (time × price_levels × features)
  - Temporal kernels of size 5, 10, 20 to capture different timescale patterns
  - ~13M params (large, not the 435K version that failed)

Architecture 3: CNN + Causal Transformer (CausalTransformerCNN)
  - CNN per-frame embeddings → Transformer encoder with causal mask
  - Multi-head attention: 8 heads, 6 layers, 256 dim
  - Can attend to frames from 30 seconds ago
  - ~15M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from book_spatial_cnn import BookSpatialCNN


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Causal Mask + Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular boolean mask for causal attention.
    True values are positions to MASK (attend nothing to future).
    Shape: (seq_len, seq_len).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to a sequence.
    Supports sequences up to max_len=600 (5x300 safety margin).
    """
    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 1: CNN + Causal Attention
# ─────────────────────────────────────────────────────────────────────────────

class CausalAttentionCNN(nn.Module):
    """
    WiderBookSpatialCNN spatial encoder → Causal Multi-Head Attention → prediction.

    Each of 300 frames is independently processed by the spatial CNN to produce
    a 512-dim embedding. The 300 embeddings are then fed through causal MHA
    so each position can attend to all prior frames.

    Key design choices:
    - CNN weights are NOT frozen (end-to-end training allows gradients to flow
      back into the spatial encoder based on the temporal attention signal)
    - Positional encoding tells the model "this frame is 3.0s before the current"
    - Single attention layer keeps parameter count reasonable while adding
      the full temporal receptive field

    ~14M params.
    """

    def __init__(
        self,
        window_size:        int   = 300,
        cnn_embed_dim:      int   = 512,
        attn_heads:         int   = 8,
        attn_dropout:       float = 0.1,
        dropout:            float = 0.15,
        num_classes:        int   = 1,
    ):
        super().__init__()
        self.window_size   = window_size
        self.cnn_embed_dim = cnn_embed_dim

        # ---- Spatial CNN backbone (wider) ----
        # Processes each frame independently via batch-over-frames trick
        # Set num_classes=cnn_embed_dim to output the embedding, not a class score
        self.spatial_cnn = BookSpatialCNN(
            window_size=1,           # each frame processed independently
            num_levels=20,
            num_features=4,
            spatial_channels=(64, 128, 256, 512),
            temporal_channels=cnn_embed_dim,
            dropout=dropout,
            num_classes=cnn_embed_dim,
        )
        # Remove the final linear projection from the classifier (keep penultimate)
        # We want the 512-dim temporal_pool output, not the classifier logits
        # Wrap the forward to stop before classifier's last linear
        self._embed_dim = cnn_embed_dim

        # ---- Positional encoding ----
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cnn_embed_dim, max_len=window_size + 10, dropout=dropout * 0.5
        )

        # ---- Causal multi-head attention ----
        self.causal_attn = nn.MultiheadAttention(
            embed_dim=cnn_embed_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(cnn_embed_dim)

        # ---- Feed-forward projection after attention ----
        self.ffn = nn.Sequential(
            nn.Linear(cnn_embed_dim, cnn_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_embed_dim * 2, cnn_embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(cnn_embed_dim)

        # ---- Prediction head (use last frame's representation) ----
        self.head = nn.Sequential(
            nn.Linear(cnn_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"CausalAttentionCNN: window={window_size}, params={n_params:,}")

    def _frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame CNN embeddings.

        Args:
            x: (B, T, 20, 4)

        Returns:
            embeddings: (B, T, embed_dim)
        """
        B, T, L, F = x.shape
        # Reshape to (B*T, 1, L, F) — treat each frame as a window_size=1 input
        x_flat = x.reshape(B * T, 1, L, F)

        # Run through spatial CNN — output is (B*T, embed_dim) from classifier
        # Note: with num_classes=embed_dim the "classifier" IS the embedding
        emb_flat = self.spatial_cnn(x_flat)  # (B*T, embed_dim)
        return emb_flat.reshape(B, T, self.cnn_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, window_size, 20, 4)

        Returns:
            predictions: (B,) or (B, num_classes)
        """
        B, T, L, F = x.shape
        assert T == self.window_size, f"Expected T={self.window_size}, got {T}"

        # 1. Per-frame spatial embeddings: (B, T, embed_dim)
        emb = self._frame_embeddings(x)

        # 2. Add positional encoding
        emb = self.pos_enc(emb)

        # 3. Causal attention: each frame attends to all prior frames only
        causal_mask = make_causal_mask(T, emb.device)  # (T, T)
        attn_out, _ = self.causal_attn(
            emb, emb, emb,
            attn_mask=causal_mask,
            need_weights=False,
        )

        # Residual + norm
        emb = self.attn_norm(emb + attn_out)

        # 4. FFN + residual
        emb = self.ffn_norm(emb + self.ffn(emb))

        # 5. Use LAST frame's representation to predict (most information-rich)
        last_frame = emb[:, -1, :]  # (B, embed_dim)

        # 6. Prediction head
        out = self.head(last_frame)  # (B, num_classes)
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 2: 3D CNN (Large)
# ─────────────────────────────────────────────────────────────────────────────

class ThreeDimensionalCNN(nn.Module):
    """
    3D Convolutional CNN that treats the book window as a 3D volume:
        (time_frames × price_levels × features)

    Input: (B, 300, 20, 4) → add channel dim → (B, 1, 300, 20, 4)

    Temporal kernels of size 5, 10, 20 detect patterns at different timescales:
        - 5-frame kernels: 0.5s patterns
        - 10-frame kernels: 1.0s patterns
        - 20-frame kernels: 2.0s patterns

    Large model: 64→128→256 channels, ~13M params.
    """

    def __init__(
        self,
        window_size: int   = 300,
        num_levels:  int   = 20,
        num_features: int  = 4,
        dropout:     float = 0.15,
        num_classes: int   = 1,
    ):
        super().__init__()
        self.window_size = window_size

        # ---- 3D conv block helper ----
        def block3d(in_ch, out_ch, t_kernel, stride=(1,1,1), padding=None):
            if padding is None:
                padding = (t_kernel // 2, 1, 0)  # causal padding added in forward
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch,
                          kernel_size=(t_kernel, 3, num_features if in_ch == 1 else 1),
                          stride=stride,
                          padding=(t_kernel // 2, 1, 0),
                          bias=False),
                nn.BatchNorm3d(out_ch),
                nn.GELU(),
                nn.Dropout3d(dropout * 0.5),
            )

        # ---- Multi-scale temporal stem ----
        # Three parallel streams with different temporal receptive fields
        self.stream_5  = block3d(1, 32, t_kernel=5)     # 0.5s patterns
        self.stream_10 = block3d(1, 32, t_kernel=10)    # 1.0s patterns
        self.stream_20 = block3d(1, 32, t_kernel=20)    # 2.0s patterns

        # Fuse 3 streams (3*32 = 96 channels after concat)
        fused_ch = 96
        self.fuse = nn.Sequential(
            nn.Conv3d(fused_ch, 64, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )

        # ---- Deeper 3D processing ----
        self.block1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(5, 3, 1), padding=(2, 1, 0), bias=False),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Dropout3d(dropout * 0.5),
        )
        # Temporal downsampling: stride=2 along time dim
        self.block2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(5, 3, 1), stride=(2, 1, 1),
                      padding=(2, 1, 0), bias=False),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(5, 3, 1), stride=(2, 1, 1),
                      padding=(2, 1, 0), bias=False),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

        # ---- Global average pooling → head ----
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ThreeDimensionalCNN: window={window_size}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 20, 4) book window

        Returns:
            predictions: (B,)
        """
        B, T, L, F = x.shape
        # Add channel dim: (B, 1, T, L, F)
        x5 = x.unsqueeze(1)

        # Multi-scale streams
        s5  = self.stream_5(x5)     # (B, 32, T, L', 1) — feature dim collapsed
        s10 = self.stream_10(x5)
        s20 = self.stream_20(x5)

        # Align temporal dims (streams with larger kernels may differ by a frame)
        # Use smallest temporal dim as reference
        t_ref = min(s5.shape[2], s10.shape[2], s20.shape[2])
        s5  = s5[:, :, :t_ref]
        s10 = s10[:, :, :t_ref]
        s20 = s20[:, :, :t_ref]

        fused = torch.cat([s5, s10, s20], dim=1)  # (B, 96, T, L', 1)
        fused = self.fuse(fused)                   # (B, 64, T, L', 1)

        out = self.block1(fused)   # (B, 128, T, L', 1)
        out = self.block2(out)     # (B, 256, T/2, L', 1)
        out = self.block3(out)     # (B, 256, T/4, L', 1)
        out = self.global_pool(out)  # (B, 256, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (B, 256)

        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 3: CNN + Causal Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CausalTransformerCNN(nn.Module):
    """
    CNN per-frame embeddings → Causal Transformer Encoder → prediction.

    The Transformer is a proper multi-layer encoder with causal masking:
        - 6 layers of multi-head attention (8 heads each)
        - 512 model dimension, 2048 FFN dim
        - Causal mask: each frame can only see prior frames
        - Pre-norm (LayerNorm before attention, more stable training)

    This is the most expressive of the three architectures. The Transformer
    can learn COMPLEX temporal dependencies — "if frame 100 looks like X
    AND frames 150-200 looked like Y, then move at frame 300."

    ~15M params.
    """

    def __init__(
        self,
        window_size:      int   = 300,
        cnn_embed_dim:    int   = 512,
        transformer_dim:  int   = 256,
        n_heads:          int   = 8,
        n_layers:         int   = 6,
        ffn_dim:          int   = 1024,
        dropout:          float = 0.15,
        attn_dropout:     float = 0.1,
        num_classes:      int   = 1,
    ):
        super().__init__()
        self.window_size = window_size

        # ---- Spatial CNN backbone ----
        self.spatial_cnn = BookSpatialCNN(
            window_size=1,
            num_levels=20,
            num_features=4,
            spatial_channels=(64, 128, 256, 512),
            temporal_channels=cnn_embed_dim,
            dropout=dropout,
            num_classes=cnn_embed_dim,
        )

        # ---- Project CNN embeddings to transformer dim ----
        self.input_proj = nn.Sequential(
            nn.Linear(cnn_embed_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ---- Positional encoding ----
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=transformer_dim, max_len=window_size + 10, dropout=dropout * 0.5
        )

        # ---- Causal Transformer layers ----
        # Pre-norm for stable training of deeper networks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-norm: more stable for deep transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,   # disable for causal masking
        )

        # ---- Prediction head ----
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"CausalTransformerCNN: window={window_size}, transformer_dim={transformer_dim}, "
              f"n_layers={n_layers}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 20, 4)

        Returns:
            predictions: (B,)
        """
        B, T, L, F = x.shape
        assert T == self.window_size

        # 1. Per-frame embeddings via spatial CNN
        x_flat  = x.reshape(B * T, 1, L, F)
        emb_flat = self.spatial_cnn(x_flat)        # (B*T, cnn_embed_dim)
        emb      = emb_flat.reshape(B, T, -1)       # (B, T, cnn_embed_dim)

        # 2. Project to transformer dimension
        emb = self.input_proj(emb)                  # (B, T, transformer_dim)

        # 3. Positional encoding
        emb = self.pos_enc(emb)

        # 4. Causal Transformer encoder
        causal_mask = make_causal_mask(T, emb.device)  # (T, T)
        out = self.transformer(emb, mask=causal_mask)   # (B, T, transformer_dim)

        # 5. Use last frame for prediction
        last_frame = out[:, -1, :]  # (B, transformer_dim)
        return self.head(last_frame).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_model(arch: str, window_size: int = 300, **kwargs):
    """
    Build one of the three temporal architectures.

    Args:
        arch: 'causal_attn' | '3d_cnn' | 'causal_transformer'
        window_size: number of frames (default 300 = 30s at 100ms)
    """
    if arch == 'causal_attn':
        return CausalAttentionCNN(window_size=window_size, **kwargs)
    elif arch == '3d_cnn':
        return ThreeDimensionalCNN(window_size=window_size, **kwargs)
    elif arch == 'causal_transformer':
        return CausalTransformerCNN(window_size=window_size, **kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}. Use causal_attn|3d_cnn|causal_transformer")


if __name__ == '__main__':
    # Architecture verification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nVerifying temporal models on {device}")
    print("="*60)

    B, T, L, F = 2, 300, 20, 4
    x = torch.randn(B, T, L, F).to(device)

    for arch in ['causal_attn', '3d_cnn', 'causal_transformer']:
        model = build_temporal_model(arch).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"  {arch}: input={tuple(x.shape)} → output={tuple(out.shape)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("All architectures OK.")
