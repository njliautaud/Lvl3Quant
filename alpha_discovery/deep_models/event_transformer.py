"""
Event Transformer — Processes tokenized MBO event sequences.

Input: NPZ files from `lob_cache_builder --mode events`
  - event_sequences: (n_bars, 200, 5) int16
    Fields per event: [event_type, price_level, size_bucket, time_delta_ms, sequence_position]
  - sequence_lengths: (n_bars,) uint16
  - mid_prices: (n_bars,) float64

Architecture (v2 — hybrid CNN + Transformer):
  - Learned embeddings for each discrete token field (event_type, price_level, size_bucket)
  - Continuous encoding for time_delta and sequence_position
  - Local 1D depthwise CNN (kernel=5) to capture short-range event patterns before attention
  - 2-layer Transformer encoder (d_model=192, nhead=6) — leaner but wider than v1
  - Mean pooling over non-padded positions
  - Optional: inject engineered features (MBO scalar features) at the pooled representation
  - FC head -> direction prediction

Key changes vs v1:
  - Added local CNN stem before transformer (captures burst patterns, reduces quadratic cost)
  - d_model 128->192, nhead 8->6 (head_dim 16->32 — more expressive per head)
  - num_layers 4->2 (fewer layers, rely on CNN stem for local; transformer for global)
  - Optional engineered_dim for hybrid input (raw events + hand-crafted features)
  - max_seq_len extended to 400 (see --event-window-bars flag in train_walkforward.py)
  - Fixed: double positional encoding removed
  - Fixed: time_delta normalization /100 -> /1000

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

class EventTokenDataset(torch.utils.data.Dataset):
    """
    Loads event token NPZ files and creates (input, target) pairs.

    Args:
        npz_dir: Directory containing YYYY-MM-DD_event_tokens.npz files
        horizon: Number of bars forward for target computation (default 20 = 2s)
        threshold: Minimum absolute return (in ticks) for directional label
    """
    def __init__(
        self,
        npz_dir: str,
        horizon: int = 20,
        threshold: float = 0.5,
    ):
        self.horizon = horizon
        self.threshold = threshold

        # Load all NPZ files and concatenate
        npz_files = sorted(Path(npz_dir).glob('*_event_tokens.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No event token NPZ files found in {npz_dir}")

        all_sequences = []
        all_lengths = []
        all_mids = []

        for f in npz_files:
            data = np.load(f)
            all_sequences.append(data['event_sequences'])
            all_lengths.append(data['sequence_lengths'])
            all_mids.append(data['mid_prices'])

        self.sequences = np.concatenate(all_sequences, axis=0)  # (N, 200, 5)
        self.lengths = np.concatenate(all_lengths, axis=0)       # (N,)
        self.mids = np.concatenate(all_mids, axis=0)             # (N,)

        # Compute targets: direction of mid price change
        # target[i] = sign(mid[i+horizon] - mid[i])
        # 0=down, 1=flat, 2=up
        n = len(self.mids)
        self.targets = np.ones(n, dtype=np.int64)  # default: flat
        for i in range(n - horizon):
            delta = self.mids[i + horizon] - self.mids[i]
            delta_ticks = delta / 0.25  # ES tick size
            if delta_ticks > threshold:
                self.targets[i] = 2  # up
            elif delta_ticks < -threshold:
                self.targets[i] = 0  # down

        # Valid indices (must have future data for target)
        self.valid_indices = list(range(n - horizon))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        seq = torch.from_numpy(self.sequences[i].astype(np.int64))   # (200, 5)
        length = int(self.lengths[i])
        target = self.targets[i]
        return seq, length, target


# =============================================================================
# Model
# =============================================================================

class EventTransformer(nn.Module):
    """
    Hybrid CNN + Transformer model for MBO event token sequences.

    v2 architecture adds a local CNN stem before the Transformer to capture
    short-range event burst patterns efficiently. The Transformer layers then
    handle long-range dependencies with lower computational cost (fewer layers).

    Architecture:
        1. Embedding layers for discrete fields (event_type, price_level, size_bucket)
        2. Linear projection for continuous fields (time_delta, sequence_position)
        3. Concatenation + linear projection to d_model
        4. Local CNN stem: depthwise Conv1d (kernel=5) for burst pattern detection
        5. 2-layer Transformer encoder (d_model=192, nhead=6, head_dim=32)
        6. Mean pooling over non-padded positions
        7. Optional engineered feature injection (engineered_dim > 0)
        8. Classification head

    Args:
        d_model: Transformer hidden dimension (default: 192)
        nhead: Number of attention heads (default: 6; head_dim = d_model/nhead = 32)
        num_layers: Number of Transformer encoder layers (default: 2)
        dim_feedforward: FFN dimension (default: 768)
        dropout: Dropout rate (default: 0.1)
        num_classes: Output classes (default: 3 for down/flat/up)
        max_seq_len: Maximum sequence length (default: 400)
        engineered_dim: If >0, also accept a vector of engineered features and
                        fuse them with the event encoding before the head (default: 0)
        use_cnn_stem: If True, add local Conv1d before Transformer (default: True)
    """
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 2,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 400,
        engineered_dim: int = 0,
        use_cnn_stem: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.engineered_dim = engineered_dim
        self.use_cnn_stem = use_cnn_stem

        # Embeddings for discrete token fields
        self.event_type_emb = nn.Embedding(8, 32)       # 8 event types -> 32d
        self.price_level_emb = nn.Embedding(21, 32)      # -10..+10 (21 levels) -> 32d
        self.size_bucket_emb = nn.Embedding(5, 16)        # 5 buckets -> 16d

        # Linear encoding for continuous fields
        self.time_delta_proj = nn.Linear(1, 16)           # time_delta_ms -> 16d
        self.seq_pos_proj = nn.Linear(1, 16)              # sequence_position -> 16d

        # Project concatenated embeddings to d_model
        # Total: 32 + 32 + 16 + 16 + 16 = 112
        self.input_proj = nn.Linear(112, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Local CNN stem: depthwise separable Conv1d across the sequence
        # Captures local burst patterns (e.g., rapid fire cancels, aggressive sweeps)
        # before the global Transformer attention.
        if use_cnn_stem:
            self.cnn_stem = nn.Sequential(
                # Depthwise conv: each channel separately (efficient, group=d_model)
                nn.Conv1d(
                    d_model, d_model, kernel_size=5, padding=2,
                    groups=d_model, bias=False
                ),
                # Pointwise conv: mix channels
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                nn.LayerNorm([d_model, 1]),  # placeholder; we do it differently below
            )
            # Replace with a cleaner implementation
            self.cnn_stem = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            )
            self.cnn_norm = nn.LayerNorm(d_model)
        else:
            self.cnn_stem = None
            self.cnn_norm = None

        # Transformer encoder (leaner: 2 layers, wider d_model/heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Optional engineered feature fusion
        if engineered_dim > 0:
            self.eng_proj = nn.Sequential(
                nn.Linear(engineered_dim, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 2),
            )
            head_in_dim = d_model + d_model // 2
        else:
            self.eng_proj = None
            head_in_dim = d_model

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        sequences: torch.Tensor,                          # (batch, seq_len, 5) int64
        sequence_lengths: torch.Tensor,                   # (batch,) int
        engineered_features: Optional[torch.Tensor] = None,  # (batch, engineered_dim) float
    ) -> torch.Tensor:
        """
        Args:
            sequences: Event token sequences (batch, max_seq_len, 5)
                       Fields: [event_type, price_level, size_bucket, time_delta_ms, seq_pos]
            sequence_lengths: Actual lengths before padding (batch,)
            engineered_features: Optional hand-crafted feature vector (batch, engineered_dim)
                                 Only used if engineered_dim > 0 was set at construction.

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = sequences.shape

        # Extract and embed each field
        event_type = sequences[:, :, 0].clamp(0, 7)          # (B, L)
        price_level = (sequences[:, :, 1] + 10).clamp(0, 20)  # shift -10..+10 to 0..20
        size_bucket = sequences[:, :, 2].clamp(0, 4)          # (B, L)
        time_delta = sequences[:, :, 3].float().unsqueeze(-1) / 1000.0  # normalize ms to seconds (50-500ms -> 0.05-0.5)
        seq_pos = sequences[:, :, 4].float().unsqueeze(-1) / 200.0     # normalize to [0,1]

        # Embeddings
        et_emb = self.event_type_emb(event_type)      # (B, L, 32)
        pl_emb = self.price_level_emb(price_level)     # (B, L, 32)
        sb_emb = self.size_bucket_emb(size_bucket)     # (B, L, 16)
        td_emb = self.time_delta_proj(time_delta)       # (B, L, 16)
        sp_emb = self.seq_pos_proj(seq_pos)             # (B, L, 16)

        # Concatenate all embeddings
        x = torch.cat([et_emb, pl_emb, sb_emb, td_emb, sp_emb], dim=-1)  # (B, L, 112)

        # Project to d_model
        x = self.input_proj(x)       # (B, L, d_model)
        x = self.input_norm(x)

        # Local CNN stem (before transformer; residual add)
        if self.cnn_stem is not None:
            # CNN operates on (B, d_model, L) — channel-first for Conv1d
            x_cnn = self.cnn_stem(x.permute(0, 2, 1))   # (B, d_model, L)
            x_cnn = x_cnn.permute(0, 2, 1)              # (B, L, d_model)
            x = self.cnn_norm(x + x_cnn)                # residual + layer norm

        # Create attention mask for padding
        # True = ignore this position
        mask = torch.arange(seq_len, device=sequences.device).unsqueeze(0)  # (1, L)
        mask = mask >= sequence_lengths.unsqueeze(1)  # (B, L) True where padded

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, L, d_model)

        # Mean pooling over non-padded positions
        mask_float = (~mask).float().unsqueeze(-1)  # (B, L, 1) 1.0 for real, 0.0 for padded
        x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        # Optional engineered feature fusion
        if self.eng_proj is not None and engineered_features is not None:
            eng = self.eng_proj(engineered_features)  # (B, d_model//2)
            x = torch.cat([x, eng], dim=-1)           # (B, d_model + d_model//2)

        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        return logits


# =============================================================================
# Training (TODO)
# =============================================================================

def train_event_transformer(
    data_dir: str,
    output_dir: str = './models/event_transformer',
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    horizon: int = 20,
    device: str = 'cuda',
) -> Dict:
    """
    Train the EventTransformer model.

    TODO: Implement full training loop with:
    - Train/val/test split by date (temporal, no leakage)
    - Learning rate scheduling (cosine annealing with warmup)
    - Early stopping on validation loss
    - Gradient clipping
    - Mixed precision training (torch.amp)
    - Checkpointing best model
    - Logging metrics to tensorboard/wandb
    - Class weight balancing (direction labels are imbalanced)

    Args:
        data_dir: Directory with *_event_tokens.npz files
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        horizon: Prediction horizon in bars
        device: 'cuda' or 'cpu'

    Returns:
        Dict with training metrics
    """
    raise NotImplementedError(
        "Training loop not yet implemented. "
        "See architecture above for model definition. "
        "Key considerations:\n"
        "  1. Split data by DATE (temporal split, no shuffle)\n"
        "  2. Use weighted cross-entropy (direction labels are imbalanced)\n"
        "  3. Apply gradient clipping (max_norm=1.0)\n"
        "  4. Use cosine annealing LR schedule with warmup\n"
        "  5. Save best model by validation accuracy\n"
    )


if __name__ == '__main__':
    # Quick architecture verification
    model = EventTransformer()
    print(f"EventTransformer parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 200
    sequences = torch.randint(0, 8, (batch_size, seq_len, 5))
    lengths = torch.randint(10, seq_len, (batch_size,))

    logits = model(sequences, lengths)
    print(f"Input shape:  ({batch_size}, {seq_len}, 5)")
    print(f"Output shape: {logits.shape}")  # Should be (4, 3)
    print(f"Logits: {logits}")

    # Test with engineered features
    model_hybrid = EventTransformer(engineered_dim=64)
    print(f"\nEventTransformer (hybrid, engineered_dim=64) parameters: "
          f"{sum(p.numel() for p in model_hybrid.parameters()):,}")
    eng_feats = torch.randn(batch_size, 64)
    logits_hybrid = model_hybrid(sequences, lengths, eng_feats)
    print(f"Hybrid output shape: {logits_hybrid.shape}")
