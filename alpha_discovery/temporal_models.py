"""
Temporal Model Progression for MBO Alpha Discovery.

Three architectures that capture sequential patterns in order book data:
1. TemporalCNN — 1D causal convolutions (fast, local patterns)
2. SequenceLSTM — Bidirectional LSTM with attention (memory, regime changes)
3. MicroTransformer — Self-attention on book sequences (global patterns)

All models:
- Input: (batch, seq_len, n_features) — windowed flat features from mbo_features.py
- Output: (batch, 1) — scalar prediction (return, volatility, etc.)
- Support variable sequence lengths
- Use LayerNorm for stable training
- Include dropout for regularization

Design Philosophy:
- Keep models SMALL — we have limited data (~50-95 trading days)
- Prefer regularization over capacity
- Causal-only where possible (no future leakage)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    1D Causal CNN for order book sequences.

    Architecture:
    - Stack of 1D causal convolutions with increasing dilation
    - Each layer: Conv1d -> LayerNorm -> GELU -> Dropout
    - Residual connections where dimensions match
    - Global average pooling -> prediction head

    Why causal convolutions:
    - Only look backward in time (no future leakage)
    - Capture local temporal patterns (microbursts, momentum)
    - Dilated convolutions give exponentially growing receptive field
    - Fast training and inference

    Receptive field with 4 layers, kernel=3, dilations [1,2,4,8]:
    = 1 + sum((k-1)*d for d in [1,2,4,8]) = 1 + 2*(1+2+4+8) = 31 bars = 3.1 seconds
    """

    def __init__(
        self,
        n_features: int = 149,
        hidden_dim: int = 128,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Causal conv layers with exponential dilation
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            # Causal padding: (kernel_size - 1) * dilation on the left only
            padding = (kernel_size - 1) * dilation
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_dim, hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,  # We'll trim the right side
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) boolean mask for valid positions

        Returns:
            (batch, n_outputs)
        """
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Conv layers expect (batch, channels, seq_len)
        h = h.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        for conv, ln in zip(self.conv_layers, self.layer_norms):
            residual = h
            h = conv(h)
            # Trim right side for causal padding
            h = h[:, :, :residual.shape[2]]
            # LayerNorm on feature dim (transpose back and forth)
            h = h.transpose(1, 2)  # (batch, seq_len, hidden_dim)
            h = ln(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h.transpose(1, 2)  # (batch, hidden_dim, seq_len)
            # Residual
            h = h + residual

        # Back to (batch, seq_len, hidden_dim)
        h = h.transpose(1, 2)

        # Pool: use last timestep (causal) or masked mean
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Use last timestep (most causal)
            h = h[:, -1, :]  # (batch, hidden_dim)

        return self.head(h)  # (batch, n_outputs)


class SequenceLSTM(nn.Module):
    """
    LSTM with temporal attention for order book sequences.

    Architecture:
    - Input projection with LayerNorm
    - Multi-layer LSTM (unidirectional for causality)
    - Temporal attention: learned query attends over LSTM outputs
    - Prediction head with residual connection

    Why LSTM:
    - Explicit memory cell captures regime changes
    - Gating mechanism handles noisy financial data well
    - Proven on sequence prediction tasks
    - Attention focuses on most informative timesteps

    Parameter budget: ~200K params (conservative for 50-95 days of data)
    """

    def __init__(
        self,
        n_features: int = 149,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_outputs: int = 1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # LSTM (unidirectional for causal inference)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,  # Causal only
        )

        # Temporal attention
        if use_attention:
            self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.attention_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attention_scale = math.sqrt(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long-term memory)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) boolean mask

        Returns:
            (batch, n_outputs)
        """
        batch_size = x.shape[0]

        # Project input
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        if self.use_attention:
            # Temporal attention: learned query attends over all timesteps
            query = self.attention_query.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
            keys = self.attention_proj(lstm_out)  # (batch, seq_len, hidden_dim)

            # Scaled dot-product attention
            attn_scores = torch.bmm(query, keys.transpose(1, 2)) / self.attention_scale
            # (batch, 1, seq_len)

            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, seq_len)
            context = torch.bmm(attn_weights, lstm_out)  # (batch, 1, hidden_dim)
            h = context.squeeze(1)  # (batch, hidden_dim)
        else:
            # Use last hidden state
            h = lstm_out[:, -1, :]  # (batch, hidden_dim)

        h = self.layer_norm(h)

        return self.head(h)  # (batch, n_outputs)


class MicroTransformer(nn.Module):
    """
    Lightweight Transformer for order book sequences.

    Architecture:
    - Input projection with positional encoding
    - Small transformer encoder (2-4 layers, 4 heads)
    - Causal attention mask (only attend to past)
    - Learned [CLS] token for sequence-level prediction

    Why Transformer:
    - Self-attention finds non-obvious cross-temporal dependencies
    - Multi-head attention captures different pattern types simultaneously
    - Position encoding preserves temporal ordering
    - Most powerful architecture for sequence modeling

    Key design choices:
    - SMALL: 2-4 layers, 4 heads, 128 dim (~300K params)
    - CAUSAL: attention mask prevents future information leakage
    - [CLS] token: dedicated classification/regression token
    - Pre-norm: more stable training than post-norm
    """

    def __init__(
        self,
        n_features: int = 149,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 200,
        n_outputs: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len + 1, d_model) * 0.02  # +1 for CLS
        )

        # Transformer encoder with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_causal_mask(self, seq_len, device):
        """
        Create causal attention mask.
        CLS token (position 0) can attend to all positions.
        Feature tokens can only attend to CLS and past tokens.
        """
        # Upper triangular = positions that should be masked (set to -inf)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()
        # CLS token (row 0) can attend to everything
        mask[0, :] = False
        return mask

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) boolean mask for valid positions

        Returns:
            (batch, n_outputs)
        """
        batch_size, seq_len, _ = x.shape

        # Project input features
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        h = torch.cat([cls_tokens, h], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        h = h + self.pos_embedding[:, :seq_len + 1, :]
        h = self.dropout(h)

        # Create attention mask
        total_len = seq_len + 1
        if self.causal:
            attn_mask = self._get_causal_mask(total_len, x.device)
        else:
            attn_mask = None

        # Key padding mask (if provided)
        if mask is not None:
            # Add True for CLS token (always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = ~torch.cat([cls_mask, mask], dim=1)  # Invert: True = masked
        else:
            key_padding_mask = None

        # Transformer encoding
        h = self.transformer(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)

        # Extract CLS token output
        cls_output = h[:, 0, :]  # (batch, d_model)
        cls_output = self.final_norm(cls_output)

        return self.head(cls_output)  # (batch, n_outputs)


class SpatialTemporalCNN(nn.Module):
    """
    CNN that processes BOTH spatial (book levels) and temporal dimensions.

    Input: node_features (batch, seq_len, 20, 9) + global_features (batch, seq_len, 45)

    Architecture:
    - Spatial encoder: 1D conv across book levels (captures depth patterns)
    - Temporal encoder: 1D causal conv across time (captures flow patterns)
    - Fusion: concatenate spatial+temporal+global -> prediction head

    This captures the full structure of the order book:
    - Spatial: how liquidity is distributed across price levels
    - Temporal: how that distribution evolves over time
    """

    def __init__(
        self,
        node_dim: int = 9,
        n_levels: int = 20,
        global_dim: int = 45,
        spatial_hidden: int = 32,
        temporal_hidden: int = 64,
        n_temporal_layers: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ):
        super().__init__()

        # Spatial encoder: process each timestep's book snapshot
        # Input: (batch*seq_len, n_levels, node_dim) -> (batch*seq_len, spatial_out)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(node_dim, spatial_hidden, kernel_size=3, padding=1),
            nn.LayerNorm([spatial_hidden, n_levels]),
            nn.GELU(),
            nn.Conv1d(spatial_hidden, spatial_hidden, kernel_size=3, padding=1),
            nn.LayerNorm([spatial_hidden, n_levels]),
            nn.GELU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)  # -> (batch*seq_len, spatial_hidden, 1)

        # Global features projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, spatial_hidden),
            nn.LayerNorm(spatial_hidden),
            nn.GELU(),
        )

        # Fusion dimension = spatial_hidden + spatial_hidden (from global proj)
        fusion_dim = spatial_hidden * 2

        # Temporal encoder: causal convolutions over fused features
        self.temporal_proj = nn.Sequential(
            nn.Linear(fusion_dim, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
            nn.GELU(),
        )

        self.temporal_convs = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        for i in range(n_temporal_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation
            self.temporal_convs.append(
                nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3,
                          dilation=dilation, padding=padding)
            )
            self.temporal_norms.append(nn.LayerNorm(temporal_hidden))

        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden // 2, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features, global_features):
        """
        Args:
            node_features: (batch, seq_len, n_levels, node_dim)
            global_features: (batch, seq_len, global_dim)

        Returns:
            (batch, n_outputs)
        """
        batch_size, seq_len, n_levels, node_dim = node_features.shape

        # Spatial encoding: process each timestep independently
        # Reshape to (batch*seq_len, node_dim, n_levels) for Conv1d
        nodes_flat = node_features.reshape(batch_size * seq_len, n_levels, node_dim)
        nodes_flat = nodes_flat.transpose(1, 2)  # (B*T, node_dim, n_levels)

        spatial = self.spatial_conv(nodes_flat)  # (B*T, spatial_hidden, n_levels)
        spatial = self.spatial_pool(spatial).squeeze(-1)  # (B*T, spatial_hidden)
        spatial = spatial.reshape(batch_size, seq_len, -1)  # (B, T, spatial_hidden)

        # Global features
        global_h = self.global_proj(global_features)  # (B, T, spatial_hidden)

        # Fuse spatial + global
        fused = torch.cat([spatial, global_h], dim=-1)  # (B, T, fusion_dim)

        # Temporal encoding
        h = self.temporal_proj(fused)  # (B, T, temporal_hidden)
        h = h.transpose(1, 2)  # (B, temporal_hidden, T)

        for conv, ln in zip(self.temporal_convs, self.temporal_norms):
            residual = h
            h = conv(h)
            h = h[:, :, :residual.shape[2]]  # Trim for causal padding
            h = h.transpose(1, 2)  # (B, T, temporal_hidden)
            h = ln(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h.transpose(1, 2)  # (B, temporal_hidden, T)
            h = h + residual

        # Use last timestep (causal)
        h = h[:, :, -1]  # (B, temporal_hidden)

        return self.head(h)  # (B, n_outputs)


# ============================================================================
# Model factory
# ============================================================================

def create_model(
    model_type: str,
    n_features: int = 149,
    n_outputs: int = 1,
    model_size: str = 'small',  # 'small', 'medium', 'large'
    **kwargs,
) -> nn.Module:
    """
    Factory function to create temporal models.

    Args:
        model_type: 'cnn', 'lstm', 'transformer', 'spatial_cnn'
        n_features: number of input features (149 for flat features)
        n_outputs: number of output predictions
        model_size: controls model capacity

    Returns:
        nn.Module
    """
    # Size presets (conservative for limited data)
    size_configs = {
        'small': {
            'hidden_dim': 64,
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'dim_feedforward': 128,
            'dropout': 0.3,
        },
        'medium': {
            'hidden_dim': 128,
            'd_model': 128,
            'n_layers': 3,
            'n_heads': 4,
            'dim_feedforward': 256,
            'dropout': 0.2,
        },
        'large': {
            'hidden_dim': 256,
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'dim_feedforward': 512,
            'dropout': 0.15,
        },
    }

    cfg = size_configs.get(model_size, size_configs['medium'])
    cfg.update(kwargs)

    if model_type == 'cnn':
        return TemporalCNN(
            n_features=n_features,
            hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'],
            dropout=cfg['dropout'],
            n_outputs=n_outputs,
        )
    elif model_type == 'lstm':
        return SequenceLSTM(
            n_features=n_features,
            hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'],
            dropout=cfg['dropout'],
            n_outputs=n_outputs,
        )
    elif model_type == 'transformer':
        return MicroTransformer(
            n_features=n_features,
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            n_outputs=n_outputs,
        )
    elif model_type == 'spatial_cnn':
        return SpatialTemporalCNN(
            n_outputs=n_outputs,
            dropout=cfg['dropout'],
            **{k: v for k, v in kwargs.items()
               if k in ['node_dim', 'n_levels', 'global_dim',
                         'spatial_hidden', 'temporal_hidden']},
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from: cnn, lstm, transformer, spatial_cnn")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
