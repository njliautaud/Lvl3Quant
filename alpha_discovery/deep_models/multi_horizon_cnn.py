"""
Multi-Horizon BookSpatialCNN — Shared backbone with 4 parallel prediction heads.

Wraps the existing BookSpatialCNN, replacing its single classifier head with
4 independent regression heads that predict mfe_net at different time horizons:

    Head 0: 100 bars  (10 seconds)
    Head 1: 300 bars  (30 seconds)
    Head 2: 600 bars  (1 minute)
    Head 3: 3000 bars (5 minutes)

The shared backbone (spatial ResBlocks + temporal ResBlocks + global avg pool)
learns a unified representation of the order book microstructure. Each head
then specialises that representation to a specific prediction horizon.

Training uses a weighted MSE loss summed across all heads:
    loss = w_10s * MSE(pred_10s, tgt_10s)
         + w_30s * MSE(pred_30s, tgt_30s)
         + w_1m  * MSE(pred_1m,  tgt_1m)
         + w_5m  * MSE(pred_5m,  tgt_5m)

The shorter horizons receive higher weight because they have lower noise and
the model's spatial features are most informative at high frequency.

Usage:
    model = MultiHorizonCNN()                              # default 4 horizons
    model = MultiHorizonCNN(horizons=[100, 600])           # only 10s + 1min
    model = MultiHorizonCNN(spatial_channels=(64, 128, 256, 512))  # wider

    # Forward pass returns dict {horizon_bars: prediction_tensor}
    preds = model(x)   # x: (B, window_size, 20, 4)
    # preds = {100: (B,), 300: (B,), 600: (B,), 3000: (B,)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from book_spatial_cnn import BookSpatialCNN


# =============================================================================
# Prediction Head
# =============================================================================

class HorizonHead(nn.Module):
    """
    A single prediction head for one time horizon.

    Architecture: Linear -> ReLU -> Dropout -> Linear -> scalar output
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) temporal feature vector

        Returns:
            (batch,) scalar prediction for this horizon
        """
        return self.head(x).squeeze(-1)


class AttentionHorizonHead(nn.Module):
    """
    Prediction head with self-attention over backbone features.

    Each horizon head learns to ATTEND to different parts of the backbone output.
    The 10s head might focus on spatial/depth features while the 5min head
    focuses on temporal/queue-age features.

    Architecture:
      1. Reshape flat features into a sequence of feature groups
      2. Self-attention lets the head select which feature groups matter
      3. Project attended features to scalar prediction

    This addresses the multi-task dilution problem by letting each head
    actively SELECT which backbone features are relevant for its horizon.
    """
    def __init__(self, in_features: int, n_groups: int = 8, n_heads: int = 4,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        # Each group gets in_features // n_groups dimensions
        self.group_dim = in_features // n_groups
        # Pad if not evenly divisible
        self.padded_features = self.group_dim * n_groups

        # Project to padded size if needed
        self.input_proj = nn.Linear(in_features, self.padded_features) if in_features != self.padded_features else nn.Identity()

        # Self-attention over feature groups
        self.attention = nn.MultiheadAttention(
            embed_dim=self.group_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.group_dim)

        # Final projection: attended features -> scalar
        self.proj = nn.Sequential(
            nn.Linear(self.padded_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) backbone feature vector

        Returns:
            (batch,) scalar prediction
        """
        B = x.shape[0]

        # Project and reshape into feature groups: (B, n_groups, group_dim)
        x = self.input_proj(x)
        x = x.view(B, self.n_groups, self.group_dim)

        # Self-attention: each group attends to other groups
        # This lets the head learn which feature groups matter for this horizon
        attended, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attended)  # residual + norm

        # Flatten back and project to scalar
        x = x.view(B, self.padded_features)
        return self.proj(x).squeeze(-1)


# =============================================================================
# Multi-Horizon Model
# =============================================================================

class MultiHorizonCNN(nn.Module):
    """
    Multi-horizon BookSpatialCNN with shared backbone and parallel heads.

    The backbone is the full BookSpatialCNN up to (but not including) the
    classifier head. Four independent HorizonHead modules replace the single
    classifier, each predicting mfe_net at a different bar horizon.

    Args:
        horizons: List of horizon values in bars. Each gets its own head.
                  Default: [100, 300, 600, 3000] = [10s, 30s, 1min, 5min]
        head_hidden_dim: Hidden layer size in each prediction head.
        window_size: Number of consecutive book snapshots per sample.
        num_levels: Book depth levels (10 bid + 10 ask = 20).
        num_features: Features per level (price_rel, depth, orders, age = 4).
        spatial_channels: Channel progression for spatial ResBlocks.
        temporal_channels: Channel count for temporal ResBlocks.
        dropout: Dropout rate for backbone and heads.
    """

    # Horizon labels for logging / display
    HORIZON_LABELS = {
        100:  '10s',
        300:  '30s',
        600:  '1min',
        3000: '5min',
    }

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        head_hidden_dim: int = 64,
        window_size: int = 20,
        num_levels: int = 20,
        num_features: int = 4,
        spatial_channels: Tuple[int, ...] = (32, 64, 128, 256),
        temporal_channels: int = 256,
        dropout: float = 0.1,
        use_attention_heads: bool = False,
    ):
        super().__init__()

        if horizons is None:
            horizons = [100, 300, 600, 3000]
        self.horizons = sorted(horizons)
        self.use_attention_heads = use_attention_heads

        # -- Build the shared backbone using BookSpatialCNN --
        # We set num_classes=1 because we will discard the classifier head.
        self.backbone = BookSpatialCNN(
            window_size=window_size,
            num_levels=num_levels,
            num_features=num_features,
            spatial_channels=spatial_channels,
            temporal_channels=temporal_channels,
            dropout=dropout,
            num_classes=1,  # placeholder, classifier will be bypassed
        )

        # Remove the classifier — we will call backbone layers directly
        # and skip self.backbone.classifier in forward().
        # Store temporal_channels for the heads.
        self.temporal_channels = temporal_channels

        # -- Build per-horizon prediction heads --
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            if use_attention_heads:
                self.heads[str(h)] = AttentionHorizonHead(
                    in_features=temporal_channels,
                    n_groups=8,
                    n_heads=4,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout,
                )
            else:
                self.heads[str(h)] = HorizonHead(
                    in_features=temporal_channels,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout,
                )

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the shared backbone up to the temporal feature vector,
        bypassing the classifier head.

        Args:
            x: (batch, window_size, 20, 4) book snapshot window

        Returns:
            (batch, temporal_channels) feature vector
        """
        bb = self.backbone
        B, T, L, F = x.shape

        # ---- Spatial processing (per-bar) ----
        x_spatial = x.reshape(B * T, 1, L, F)
        x_spatial = bb.spatial_stem(x_spatial)
        x_spatial = bb.spatial_res_blocks(x_spatial)
        x_spatial = bb.spatial_pool(x_spatial)
        x_spatial = x_spatial.reshape(B * T, -1)
        x_spatial = bb.spatial_compress(x_spatial)
        x_spatial = x_spatial.reshape(B, T, -1)

        # ---- Bid/Ask side features ----
        bid_in = x[:, :, :10, :].reshape(B * T, 10, F).permute(0, 2, 1)
        ask_in = x[:, :, 10:, :].reshape(B * T, 10, F).permute(0, 2, 1)

        bid_feats = bb.bid_conv(bid_in).squeeze(-1)
        ask_feats = bb.ask_conv(ask_in).squeeze(-1)
        side_feats = torch.cat([bid_feats, ask_feats], dim=-1)
        side_feats = side_feats.reshape(B, T, -1)

        # ---- Temporal processing ----
        combined = torch.cat([x_spatial, side_feats], dim=-1)
        combined = combined.permute(0, 2, 1)

        temporal_out = bb.temporal_stem(combined)
        temporal_out = bb.temporal_res1(temporal_out)
        temporal_out = bb.temporal_res2(temporal_out)
        temporal_out = bb.temporal_pool(temporal_out)
        temporal_out = temporal_out.squeeze(-1)  # (B, temporal_channels)

        return temporal_out

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass through shared backbone + all horizon heads.

        Args:
            x: (batch, window_size, 20, 4) book snapshot window

        Returns:
            Dict mapping horizon (int bars) to prediction tensor (batch,).
            Example: {100: tensor([...]), 300: tensor([...]), ...}
        """
        features = self._backbone_features(x)

        predictions = {}
        for h in self.horizons:
            predictions[h] = self.heads[str(h)](features)

        return predictions

    def predict_horizon(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Predict for a single horizon only (useful at inference time).

        Args:
            x: (batch, window_size, 20, 4) book snapshot window
            horizon: which horizon head to use (e.g. 100, 300, 600, 3000)

        Returns:
            (batch,) predictions for the requested horizon
        """
        features = self._backbone_features(x)
        return self.heads[str(horizon)](features)

    def get_horizon_label(self, horizon: int) -> str:
        """Human-readable label for a horizon value."""
        return self.HORIZON_LABELS.get(horizon, f'{horizon}bars')


# =============================================================================
# Loss Function
# =============================================================================

class MultiHorizonLoss(nn.Module):
    """
    Weighted sum of per-horizon MSE losses.

    Default weights emphasise shorter horizons (lower noise, more actionable):
        10s:  1.0
        30s:  0.8
        1min: 0.6
        5min: 0.4

    Args:
        horizons: List of horizon bar counts matching model heads.
        weights: Optional dict {horizon: weight}. If None, uses defaults.
        loss_fn: Base loss function (default: HuberLoss for robustness to
                 outlier ticks, matching train_walkforward.py convention).
    """

    DEFAULT_WEIGHTS = {
        100:  1.0,
        300:  0.8,
        600:  0.6,
        3000: 0.4,
    }

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        weights: Optional[Dict[int, float]] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()

        if horizons is None:
            horizons = [100, 300, 600, 3000]
        self.horizons = horizons

        if weights is None:
            self.weights = {h: self.DEFAULT_WEIGHTS.get(h, 0.5) for h in horizons}
        else:
            self.weights = weights

        # Normalise weights so they sum to 1 (keeps loss scale interpretable)
        total_w = sum(self.weights.values())
        self.weights = {h: w / total_w for h, w in self.weights.items()}

        self.loss_fn = loss_fn or nn.HuberLoss(delta=1.0)

    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute weighted multi-horizon loss.

        Args:
            predictions: {horizon: (batch,) tensor} from model forward pass
            targets: {horizon: (batch,) tensor} ground truth per horizon

        Returns:
            (total_loss, per_horizon_losses) where per_horizon_losses is
            {horizon: float} for logging.
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        per_horizon = {}

        for h in self.horizons:
            pred = predictions[h]
            tgt = targets[h]

            # Mask out NaN targets (day-boundary bars)
            mask = torch.isfinite(tgt)
            if mask.sum() == 0:
                per_horizon[h] = 0.0
                continue

            # Force float32 for loss computation (AMP safety)
            h_loss = self.loss_fn(pred[mask].float(), tgt[mask].float())
            per_horizon[h] = h_loss.item()
            total_loss = total_loss + self.weights[h] * h_loss

        return total_loss, per_horizon


# =============================================================================
# Verification
# =============================================================================

if __name__ == '__main__':
    print('Multi-Horizon BookSpatialCNN')
    print('=' * 60)

    model = MultiHorizonCNN()
    n_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())

    print(f'Total parameters:    {n_params:,}')
    print(f'Backbone parameters: {backbone_params:,}')
    print(f'Head parameters:     {head_params:,} ({len(model.horizons)} heads)')
    print(f'Horizons: {[model.get_horizon_label(h) for h in model.horizons]}')
    print()

    # Test forward pass
    B = 4
    x = torch.randn(B, 20, 20, 4)
    preds = model(x)

    print('Forward pass:')
    print(f'  Input:  ({B}, 20, 20, 4)')
    for h, p in preds.items():
        print(f'  {model.get_horizon_label(h):>5s} head: {p.shape}  sample={p[0].item():.4f}')

    # Test loss
    print()
    criterion = MultiHorizonLoss()
    targets = {h: torch.randn(B) for h in model.horizons}
    loss, per_h = criterion(preds, targets)
    print(f'Loss: {loss.item():.4f}')
    for h, l in per_h.items():
        label = model.get_horizon_label(h)
        print(f'  {label:>5s}: {l:.4f} (weight={criterion.weights[h]:.3f})')

    # Test single-horizon inference
    print()
    single = model.predict_horizon(x, 100)
    print(f'Single-horizon (10s) prediction: {single.shape}')
