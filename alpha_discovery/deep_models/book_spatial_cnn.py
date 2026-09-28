"""
Book Spatial CNN — Processes raw 10-level order book snapshots as 2D tensors.

Input: NPZ files from `lob_cache_builder --mode book`
  - book_tensors: (n_bars, 20, 4) float32
    20 levels: [10 bid levels (best to worst), 10 ask levels (best to worst)]
    4 features per level: [price_relative_to_mid, depth_lots, num_orders, queue_age_seconds]
  - mid_prices: (n_bars,) float64

Architecture:
  - Input: (batch, window=20, 20_levels, 4_features) -- 20 consecutive bars of book state
  - Conv2D layers processing the level x feature spatial structure (with residual connections)
  - Temporal convolution across the 20-bar window (with residual connections)
  - Global average pooling
  - Classification head

The key insight is that the order book has SPATIAL structure:
  - Adjacent levels are related (depth at level 2 relates to depth at level 1)
  - Bid/ask sides mirror each other
  - The 4 features per level form a local feature vector

A 2D CNN can learn filters that detect patterns like:
  - Depth walls (large size concentrated at one level)
  - Depth thinning (gradual decrease in depth)
  - Bid/ask asymmetries
  - Queue age patterns (fresh vs stale liquidity)

Target: Direction of mid-price 20 bars (2 seconds) into the future.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, List


# =============================================================================
# Dataset
# =============================================================================

class BookTensorDataset(torch.utils.data.Dataset):
    """
    Loads book tensor NPZ files and creates windowed (input, target) pairs.

    Each sample is a window of `window_size` consecutive book snapshots.

    Args:
        npz_dir: Directory containing YYYY-MM-DD_book_tensors.npz files
        window_size: Number of consecutive bars per sample (default 20 = 2s)
        horizon: Number of bars forward for target computation (default 20)
        threshold: Minimum return in ticks for directional label
        augment: If True, apply Gaussian noise augmentation during training
    """
    def __init__(
        self,
        npz_dir: str,
        window_size: int = 20,
        horizon: int = 20,
        threshold: float = 0.5,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.threshold = threshold
        self.augment = augment

        # Load all NPZ files
        npz_files = sorted(Path(npz_dir).glob('*_book_tensors.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No book tensor NPZ files found in {npz_dir}")

        all_tensors = []
        all_mids = []
        # Track day boundaries so we don't create cross-day samples
        self.day_boundaries = [0]

        for f in npz_files:
            data = np.load(f)
            tensors = data['book_tensors']   # (n_bars, 20, 4)
            mids = data['mid_prices']         # (n_bars,)
            all_tensors.append(tensors)
            all_mids.append(mids)
            self.day_boundaries.append(self.day_boundaries[-1] + len(mids))

        self.tensors = np.concatenate(all_tensors, axis=0)  # (N, 20, 4)
        self.mids = np.concatenate(all_mids, axis=0)        # (N,)

        # Normalize features globally
        # [0] price_relative_to_mid: already in ticks, leave as-is
        # [1] depth_lots: log-transform
        # [2] num_orders: log-transform
        # [3] queue_age_seconds: log-transform
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])  # log(1 + depth)
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])  # log(1 + orders)
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])  # log(1 + age)

        # Build valid indices: must have enough history AND future
        self.valid_indices = []
        for day_idx in range(len(self.day_boundaries) - 1):
            start = self.day_boundaries[day_idx]
            end = self.day_boundaries[day_idx + 1]
            # Need window_size bars of history + horizon bars of future
            for i in range(start + window_size - 1, end - horizon):
                self.valid_indices.append(i)

        # Compute targets
        n = len(self.mids)
        self.targets = np.ones(n, dtype=np.int64)  # default: flat
        for i in range(n - horizon):
            delta_ticks = (self.mids[i + horizon] - self.mids[i]) / 0.25
            if delta_ticks > threshold:
                self.targets[i] = 2
            elif delta_ticks < -threshold:
                self.targets[i] = 0

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Window: (window_size, 20, 4)
        window = self.tensors[i - self.window_size + 1 : i + 1].copy()

        # Data augmentation: add small Gaussian noise to depth/orders/age features
        if self.augment:
            # Noise only on log-transformed features (cols 1,2,3), small relative noise
            noise = np.random.normal(0, 0.02, window[:, :, 1:].shape).astype(np.float32)
            window[:, :, 1:] = window[:, :, 1:] + noise

        window = torch.from_numpy(window.astype(np.float32))
        target = self.targets[i]
        return window, target


# =============================================================================
# Building Blocks
# =============================================================================

class SpatialResBlock(nn.Module):
    """
    Residual block for 2D spatial processing of order book.
    Uses a 1x1 projection shortcut when channel dimensions change.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)

        # Projection shortcut if channels differ
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + identity)
        return out


class TemporalResBlock(nn.Module):
    """
    Residual block for 1D temporal processing.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + identity)
        return out


# =============================================================================
# Model
# =============================================================================

class BookSpatialCNN(nn.Module):
    """
    Spatial CNN for order book snapshots with residual connections.

    Treats the order book as a 2D spatial structure:
      - Height dimension: 20 price levels (10 bid + 10 ask)
      - Width dimension: 4 features per level
      - Temporal dimension: 20 consecutive snapshots

    The model first processes each snapshot spatially (level x feature patterns)
    using residual blocks, then processes the temporal sequence of spatial features
    with temporal residual blocks.

    Architecture:
        1. Spatial ResBlocks: process each bar's (20, 4) book state
        2. Temporal ResBlocks: process the sequence of spatial features
        3. Global average pooling
        4. Classification head

    Input: (batch, window_size, 20_levels, 4_features)
    Output: (batch, num_classes)

    Args:
        window_size: Number of consecutive bars (default: 20)
        num_levels: Number of book levels (default: 20 = 10 bid + 10 ask)
        num_features: Features per level (default: 4)
        spatial_channels: Conv2D channel progression
        temporal_channels: Conv1D channels for temporal processing
        dropout: Dropout rate
        num_classes: Output classes
    """
    def __init__(
        self,
        window_size: int = 20,
        num_levels: int = 20,
        num_features: int = 4,
        spatial_channels: Tuple[int, ...] = (32, 64, 128, 256),
        temporal_channels: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_levels = num_levels
        self.num_features = num_features

        # ---- Spatial feature extractor (processes each bar's book state) ----
        # Input: (B*T, 1, 20, 4) where T=window_size
        # Stem: initial conv to bring to first channel count
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(1, spatial_channels[0], kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_channels[0]),
            nn.GELU(),
        )

        # Residual blocks for spatial processing
        spatial_res_layers = []
        for i in range(len(spatial_channels) - 1):
            spatial_res_layers.append(
                SpatialResBlock(spatial_channels[i], spatial_channels[i + 1], dropout=dropout * 0.5)
            )
        self.spatial_res_blocks = nn.Sequential(*spatial_res_layers)

        # Pool across the feature dimension (width) but keep level dimension
        self.spatial_pool = nn.AdaptiveAvgPool2d((num_levels, 1))  # -> (B*T, C, 20, 1)

        spatial_out_dim = spatial_channels[-1] * num_levels  # 256 * 20 = 5120

        # Compress spatial features
        self.spatial_compress = nn.Sequential(
            nn.Linear(spatial_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Bid/Ask specific convolutions ----
        # Process bid and ask sides separately to learn asymmetric patterns
        self.bid_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ask_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ---- Temporal processor with residual blocks ----
        # Input: (B, 256 + 64, T)
        temporal_in = 256 + 64  # spatial_compress output + bid/ask features

        # Project to temporal_channels
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(temporal_in, temporal_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
        )

        # Temporal residual blocks
        self.temporal_res1 = TemporalResBlock(temporal_channels, kernel_size=5, dropout=dropout)
        self.temporal_res2 = TemporalResBlock(temporal_channels, kernel_size=3, dropout=dropout)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, temporal_channels, 1)

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, 20, 4) book snapshot window

        Returns:
            logits: (batch, num_classes)
        """
        B, T, L, F = x.shape
        assert T == self.window_size
        assert L == self.num_levels
        assert F == self.num_features

        # ---- Process each bar spatially ----
        # Reshape to (B*T, 1, L, F) for Conv2D
        x_spatial = x.reshape(B * T, 1, L, F)
        x_spatial = self.spatial_stem(x_spatial)           # (B*T, C0, L, F)
        x_spatial = self.spatial_res_blocks(x_spatial)     # (B*T, C_last, L, F)
        x_spatial = self.spatial_pool(x_spatial)           # (B*T, C_last, L, 1)
        x_spatial = x_spatial.reshape(B * T, -1)           # (B*T, C_last*L)
        x_spatial = self.spatial_compress(x_spatial)       # (B*T, 256)
        x_spatial = x_spatial.reshape(B, T, -1)            # (B, T, 256)

        # ---- Bid/Ask specific features ----
        # x[:, :, :10, :] = bid levels, x[:, :, 10:, :] = ask levels
        # Process as (B*T, F, 10) for Conv1d along the level dimension
        bid_in = x[:, :, :10, :].reshape(B * T, 10, F).permute(0, 2, 1)  # (B*T, F, 10)
        ask_in = x[:, :, 10:, :].reshape(B * T, 10, F).permute(0, 2, 1)  # (B*T, F, 10)

        bid_feats = self.bid_conv(bid_in).squeeze(-1)  # (B*T, 32)
        ask_feats = self.ask_conv(ask_in).squeeze(-1)  # (B*T, 32)
        side_feats = torch.cat([bid_feats, ask_feats], dim=-1)  # (B*T, 64)
        side_feats = side_feats.reshape(B, T, -1)  # (B, T, 64)

        # ---- Combine and process temporally ----
        combined = torch.cat([x_spatial, side_feats], dim=-1)  # (B, T, 320)
        combined = combined.permute(0, 2, 1)  # (B, 320, T) for Conv1d

        temporal_out = self.temporal_stem(combined)    # (B, temporal_channels, T)
        temporal_out = self.temporal_res1(temporal_out)
        temporal_out = self.temporal_res2(temporal_out)
        temporal_out = self.temporal_pool(temporal_out)  # (B, temporal_channels, 1)
        temporal_out = temporal_out.squeeze(-1)           # (B, temporal_channels)

        # ---- Classify ----
        logits = self.classifier(temporal_out)  # (B, num_classes)
        return logits


# =============================================================================
# Training (TODO)
# =============================================================================

def train_book_spatial_cnn(
    data_dir: str,
    output_dir: str = './models/book_spatial_cnn',
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    window_size: int = 20,
    horizon: int = 20,
    device: str = 'cuda',
) -> Dict:
    """
    Train the BookSpatialCNN model.

    TODO: Implement full training loop with:
    - Temporal train/val/test split by date (NO SHUFFLE)
    - Data augmentation: random noise injection, level permutation
    - Learning rate scheduling (OneCycleLR)
    - Early stopping on validation loss
    - Gradient clipping
    - Mixed precision training
    - Checkpointing
    - Feature normalization (z-score per feature across training set)
    - Class weight balancing
    - Multi-horizon targets (2s, 5s, 10s) for auxiliary losses

    Args:
        data_dir: Directory with *_book_tensors.npz files
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        window_size: Number of bars per window
        horizon: Prediction horizon in bars
        device: 'cuda' or 'cpu'

    Returns:
        Dict with training metrics
    """
    raise NotImplementedError(
        "Training loop not yet implemented. "
        "See architecture above for model definition. "
        "Key considerations:\n"
        "  1. Temporal split by date (no data leakage!)\n"
        "  2. Normalize features using TRAINING set statistics only\n"
        "  3. Window stride for training (e.g., every 5 bars, not every bar)\n"
        "  4. Consider multi-horizon auxiliary losses\n"
        "  5. Book state is already spatial -- CNN filters learn level patterns\n"
        "  6. Use BookTensorDataset with augment=True for training set\n"
    )


if __name__ == '__main__':
    # Quick architecture verification
    model = BookSpatialCNN()
    print(f"BookSpatialCNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy data
    batch_size = 4
    window_size = 20
    x = torch.randn(batch_size, window_size, 20, 4)

    logits = model(x)
    print(f"Input shape:  ({batch_size}, {window_size}, 20, 4)")
    print(f"Output shape: {logits.shape}")  # Should be (4, 3)
    print(f"Logits: {logits}")
