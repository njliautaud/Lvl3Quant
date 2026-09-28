"""
Trade Sequence LSTM — Processes raw trade-by-trade flow for directional prediction.

Input: NPZ files from `lob_cache_builder --mode trades`
  - trade_sequences: (n_bars, 50, 5) float32
    50 trades per bar, 5 features:
    [aggressor_side, size_lots, price_ticks_from_mid, time_delta_ms, cumulative_volume]
  - trade_lengths: (n_bars,) uint16
  - mid_prices: (n_bars,) float64

Architecture (v2):
  - Input: (batch, window=20*max_trades, 5) -- concatenated trade sequences from 20 bars
  - Highway input projection (5 -> 128) with skip connection for gradient flow
  - 2-layer bidirectional LSTM (hidden=192 per direction)
  - LayerNorm after LSTM output
  - Attention pooling over all timesteps with dropout
  - FC head -> direction prediction (3-class: down, flat, up)

Key changes vs v1:
  - hidden_dim 128->192 (more capacity for complex trade flow patterns)
  - Highway connection in input projection (helps gradient flow through long sequences)
  - Attention dropout added (regularization)
  - Deeper classifier head with residual

The key insight is that trade flow has TEMPORAL structure:
  - Sequences of buy trades indicate aggressive buying pressure
  - Large trades at the bid indicate selling sweeps
  - Time between trades indicates urgency
  - Cumulative volume reveals acceleration/deceleration

An LSTM with attention can learn patterns like:
  - Buy sweeps (consecutive aggressive buys with decreasing inter-trade time)
  - Iceberg detection (repeated same-size trades at same price)
  - Exhaustion (large volume with price not moving)
  - Momentum ignition (accelerating trades in one direction)

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

class TradeFlowDataset(torch.utils.data.Dataset):
    """
    Loads trade flow NPZ files and creates windowed (input, target) pairs.

    Each sample concatenates trade sequences from `window_size` consecutive bars,
    creating one long trade-by-trade sequence.

    Args:
        npz_dir: Directory containing YYYY-MM-DD_trade_flow.npz files
        window_size: Number of consecutive bars to concatenate (default 20 = 2s)
        horizon: Number of bars forward for target computation (default 20)
        threshold: Minimum return in ticks for directional label
    """
    def __init__(
        self,
        npz_dir: str,
        window_size: int = 20,
        horizon: int = 20,
        threshold: float = 0.5,
        max_trades_per_bar: int = 50,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.threshold = threshold
        self.max_trades_per_bar = max_trades_per_bar

        # Load all NPZ files
        npz_files = sorted(Path(npz_dir).glob('*_trade_flow.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No trade flow NPZ files found in {npz_dir}")

        all_seqs = []
        all_lengths = []
        all_mids = []
        self.day_boundaries = [0]

        for f in npz_files:
            data = np.load(f)
            all_seqs.append(data['trade_sequences'])     # (n_bars, 50, 5)
            all_lengths.append(data['trade_lengths'])     # (n_bars,)
            all_mids.append(data['mid_prices'])           # (n_bars,)
            self.day_boundaries.append(self.day_boundaries[-1] + len(data['mid_prices']))

        self.trade_sequences = np.concatenate(all_seqs, axis=0)    # (N, 50, 5)
        self.trade_lengths = np.concatenate(all_lengths, axis=0)   # (N,)
        self.mids = np.concatenate(all_mids, axis=0)               # (N,)

        # Normalize features
        # [0] aggressor_side: already -1/+1, leave as-is
        # [1] size_lots: log-transform
        # [2] price_ticks_from_mid: already in ticks, leave as-is
        # [3] time_delta_ms: log-transform
        # [4] cumulative_volume: log-transform
        self.trade_sequences[:, :, 1] = np.log1p(np.abs(self.trade_sequences[:, :, 1]))
        self.trade_sequences[:, :, 3] = np.log1p(self.trade_sequences[:, :, 3])
        self.trade_sequences[:, :, 4] = np.log1p(self.trade_sequences[:, :, 4])

        # Build valid indices
        self.valid_indices = []
        for day_idx in range(len(self.day_boundaries) - 1):
            start = self.day_boundaries[day_idx]
            end = self.day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end - horizon):
                self.valid_indices.append(i)

        # Compute targets
        n = len(self.mids)
        self.targets = np.ones(n, dtype=np.int64)
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

        # Concatenate trade sequences from window_size bars
        # Each bar has up to max_trades_per_bar trades
        window_start = i - self.window_size + 1
        window_end = i + 1

        sequences = self.trade_sequences[window_start:window_end]  # (W, 50, 5)
        lengths = self.trade_lengths[window_start:window_end]       # (W,)

        # Flatten into one long sequence, keeping only actual trades
        all_trades = []
        for bar_idx in range(self.window_size):
            actual_len = int(lengths[bar_idx])
            if actual_len > 0:
                all_trades.append(sequences[bar_idx, :actual_len, :])

        if all_trades:
            concat_trades = np.concatenate(all_trades, axis=0)  # (total_trades, 5)
        else:
            concat_trades = np.zeros((1, 5), dtype=np.float32)

        # Pad/truncate to fixed length
        max_total = self.window_size * self.max_trades_per_bar  # 20 * 50 = 1000
        total_trades = len(concat_trades)
        if total_trades > max_total:
            concat_trades = concat_trades[:max_total]
            total_trades = max_total
        elif total_trades < max_total:
            padding = np.zeros((max_total - total_trades, 5), dtype=np.float32)
            concat_trades = np.concatenate([concat_trades, padding], axis=0)

        seq_tensor = torch.from_numpy(concat_trades.astype(np.float32))
        target = self.targets[i]
        actual_length = min(total_trades, max_total)

        return seq_tensor, actual_length, target


# =============================================================================
# Model Components
# =============================================================================

class HighwayLayer(nn.Module):
    """
    Highway network layer for better gradient flow through deep input projections.

    output = T * H(x) + (1-T) * x
    where T = sigmoid(W_T * x + b_T) is the transform gate
    and H(x) = GELU(W_H * x + b_H) is the transform function.

    Dimensions must match for the identity path; if they don't, uses a linear projection.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.H = nn.Linear(in_dim, out_dim)
        self.T = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        # Bias toward carry (1-T) path at initialization — helps initial gradient flow
        nn.init.constant_(self.T.bias, -1.0)

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = F.gelu(self.H(x))
        T = torch.sigmoid(self.T(x))
        carry = self.shortcut(x)
        out = T * H + (1 - T) * carry
        return self.dropout(out)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling: learns which timesteps are most important.

    Instead of simple mean pooling, uses a learned attention mechanism
    to weight each timestep's contribution to the final representation.

    Args:
        hidden_dim: Input dimension
        dropout: Dropout applied to attention weights (regularization)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) True for valid positions

        Returns:
            pooled: (batch, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        weights = self.attn_dropout(weights)
        pooled = (x * weights).sum(dim=1)  # (B, hidden_dim)
        return pooled


# =============================================================================
# Main Model
# =============================================================================

class TradeSequenceLSTM(nn.Module):
    """
    Bidirectional LSTM with highway input projection and attention pooling
    for trade flow sequences.

    Processes a concatenated sequence of trades from multiple bars and
    learns to predict future price direction from the flow patterns.

    Architecture (v2):
        1. Highway input projection: 5 -> 128 (better gradient flow)
        2. 2-layer bidirectional LSTM (hidden=192 per direction = 384 total)
        3. LayerNorm after LSTM output
        4. Attention pooling with dropout across all timesteps
        5. Classification head with residual

    Args:
        input_dim: Features per trade (default: 5)
        hidden_dim: LSTM hidden dimension per direction (default: 192)
        num_layers: LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.2)
        num_classes: Output classes (default: 3)
        max_seq_len: Maximum total trades in concatenated window (default: 1000)
        proj_dim: Intermediate dimension after input highway (default: 128)
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
        max_seq_len: int = 1000,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Highway input projection: 5 -> proj_dim
        # Better gradient flow than plain Linear+GELU for long sequences
        self.input_proj = HighwayLayer(input_dim, proj_dim, dropout=dropout * 0.5)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer normalization after LSTM output
        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # Attention pooling with dropout
        self.attention_pool = AttentionPooling(lstm_out_dim, dropout=dropout * 0.5)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        sequences: torch.Tensor,          # (batch, max_seq_len, 5)
        sequence_lengths: torch.Tensor,    # (batch,) int
    ) -> torch.Tensor:
        """
        Args:
            sequences: Concatenated trade sequences (batch, max_seq_len, input_dim)
            sequence_lengths: Actual sequence lengths (batch,)

        Returns:
            logits: (batch, num_classes)
        """
        B, L, D = sequences.shape

        # Highway input projection
        x = self.input_proj(sequences)  # (B, L, proj_dim)

        # Pack padded sequences for efficient LSTM processing
        lengths_cpu = sequence_lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        lstm_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=L
        )  # (B, L, hidden_dim*2)

        # Layer norm
        lstm_out = self.lstm_norm(lstm_out)

        # Create mask for attention pooling
        mask = torch.arange(L, device=sequences.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        # (B, L) True for valid positions

        # Attention pooling
        pooled = self.attention_pool(lstm_out, mask)  # (B, hidden_dim*2)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


# =============================================================================
# Training (TODO)
# =============================================================================

def train_trade_sequence_lstm(
    data_dir: str,
    output_dir: str = './models/trade_sequence_lstm',
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    window_size: int = 20,
    horizon: int = 20,
    device: str = 'cuda',
) -> Dict:
    """
    Train the TradeSequenceLSTM model.

    TODO: Implement full training loop with:
    - Temporal train/val/test split by date
    - Custom collate function for variable-length sequences
    - Learning rate scheduling (ReduceLROnPlateau)
    - Early stopping on validation loss
    - Gradient clipping (critical for LSTMs!)
    - Mixed precision training
    - Checkpointing best model
    - Trade-level data augmentation:
      * Random trade dropping (simulate missed trades)
      * Random time jitter
      * Size noise injection
    - Multi-task: predict direction + magnitude simultaneously
    - Curriculum learning: start with easy examples (large moves), add hard ones

    Args:
        data_dir: Directory with *_trade_flow.npz files
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
        "  1. Temporal split by date (no leakage)\n"
        "  2. Gradient clipping is CRITICAL for LSTMs (max_norm=1.0)\n"
        "  3. Variable-length sequences need pack_padded_sequence\n"
        "  4. Trade sequences can be very sparse (bars with 0 trades)\n"
        "  5. Consider curriculum learning for convergence\n"
        "  6. Attention weights are interpretable -- visualize them!\n"
    )


# =============================================================================
# Ensemble
# =============================================================================

class TradeFlowEnsemble(nn.Module):
    """
    Ensemble model that combines predictions from all three deep learning models.

    This is a simple weighted average ensemble. Can be extended with:
    - Learned weights (meta-learner)
    - Stacking (train a meta-model on outputs)
    - Conditional routing (use different models for different regimes)

    TODO: Implement after individual models are trained.
    """
    def __init__(
        self,
        event_model: Optional[nn.Module] = None,
        book_model: Optional[nn.Module] = None,
        trade_model: Optional[nn.Module] = None,
        weights: Tuple[float, ...] = (0.33, 0.33, 0.34),
    ):
        super().__init__()
        self.event_model = event_model
        self.book_model = book_model
        self.trade_model = trade_model
        self.weights = weights

    def forward(self, event_data=None, book_data=None, trade_data=None):
        """
        Args:
            event_data: Tuple of (sequences, lengths) for EventTransformer
            book_data: Tensor for BookSpatialCNN
            trade_data: Tuple of (sequences, lengths) for TradeSequenceLSTM

        Returns:
            ensemble_logits: (batch, num_classes)
        """
        logits_list = []
        weight_list = []

        if self.event_model is not None and event_data is not None:
            logits_list.append(self.event_model(*event_data))
            weight_list.append(self.weights[0])

        if self.book_model is not None and book_data is not None:
            logits_list.append(self.book_model(book_data))
            weight_list.append(self.weights[1])

        if self.trade_model is not None and trade_data is not None:
            logits_list.append(self.trade_model(*trade_data))
            weight_list.append(self.weights[2])

        if not logits_list:
            raise ValueError("At least one model must provide predictions")

        # Normalize weights
        total_weight = sum(weight_list)
        weight_list = [w / total_weight for w in weight_list]

        # Weighted average of probabilities (not logits)
        probs = [F.softmax(l, dim=-1) * w for l, w in zip(logits_list, weight_list)]
        ensemble_probs = sum(probs)

        return torch.log(ensemble_probs + 1e-8)  # log-probs for NLLLoss compatibility


if __name__ == '__main__':
    # Quick architecture verification
    model = TradeSequenceLSTM()
    print(f"TradeSequenceLSTM parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy data
    batch_size = 4
    max_seq_len = 1000
    sequences = torch.randn(batch_size, max_seq_len, 5)
    lengths = torch.randint(50, max_seq_len, (batch_size,))

    logits = model(sequences, lengths)
    print(f"Input shape:  ({batch_size}, {max_seq_len}, 5)")
    print(f"Output shape: {logits.shape}")  # Should be (4, 3)
    print(f"Logits: {logits}")
