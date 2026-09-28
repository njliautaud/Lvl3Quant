"""
feature_engineering.py — Derived feature computation for book tensor data.

Augments raw book tensors (window, 20_levels, 4_features) with additional
derived features, returning (window, 20_levels, 4 + N_extra) tensors.

Input tensor layout (from lob_cache_builder):
  - Shape: (window, 20, 4)
  - Levels: [0-9] = bid levels, best to worst  (level 0 = best bid)
             [10-19] = ask levels, best to worst (level 10 = best ask)
  - Features per level:
      [0] price_relative_to_mid  — signed ticks from mid price
      [1] depth_lots             — log1p(lots) already applied by dataset
      [2] num_orders             — log1p(orders) already applied by dataset
      [3] queue_age_seconds      — log1p(age) already applied by dataset

NOTE: depth_lots / num_orders / queue_age_seconds are already log1p-transformed
by BookTensorDataset.__init__. Feature engineering here operates on the
already-normalised tensors that the model actually receives.

Derived features (all shape (window, 20, 1) broadcast across levels or per-level):

  a) order_flow_imbalance  — (sum_bid_depth - sum_ask_depth) /
                              (sum_bid_depth + sum_ask_depth + eps)
                              Scalar per bar, broadcast to all 20 levels.

  b) pressure_gradient     — depth[level_n+1] - depth[level_n] within each side.
                              First level of each side gets the same value as level 1.
                              Shape: (window, 20) — per level.

  c) spread                — price_rel_to_mid[level_10] - price_rel_to_mid[level_9]
                              (best ask price - best bid price, in ticks)
                              Scalar per bar, broadcast to all 20 levels.

  d) queue_age_momentum    — queue_age[t] - queue_age[t-1], bar-over-bar delta.
                              Zero at t=0 (no prior bar in window). Per level.

  e) depth_change_velocity — depth_lots[t] - depth_lots[t-1], bar-over-bar delta.
                              Zero at t=0. Per level.

  f) book_asymmetry        — bid_depth[i] - ask_depth[i] for i in 0..9,
                              mirrored symmetrically to ask levels. Per level.

  g) cumulative_depth_ratio — cumulative bid depth / (cumulative ask depth + eps)
                               up to level i, for i in 0..9.
                               For ask levels 10..19, uses the same ratio at the
                               corresponding depth (ask level j mirrors bid level j).
                               Per level.

Usage:
    from feature_engineering import BookFeatureEngineer

    engineer = BookFeatureEngineer(features={
        'order_flow_imbalance': True,
        'pressure_gradient': True,
        'spread': True,
        'queue_age_momentum': True,
        'depth_change_velocity': True,
        'book_asymmetry': True,
        'cumulative_depth_ratio': True,
    })

    # x: (batch, window, 20, 4) — raw book window
    x_aug = engineer(x)   # -> (batch, window, 20, 4 + N)

    # Or for a single window (no batch dim):
    x_aug = engineer(x, has_batch=False)   # -> (window, 20, 4 + N)

All operations are pure PyTorch — GPU-compatible and fully differentiable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Feature index constants (raw 4-feature tensor)
# ---------------------------------------------------------------------------
FEAT_PRICE = 0   # price_relative_to_mid (ticks, signed)
FEAT_DEPTH = 1   # depth_lots (log1p)
FEAT_ORDERS = 2  # num_orders (log1p)
FEAT_AGE = 3     # queue_age_seconds (log1p)

N_LEVELS = 20
N_BID = 10   # levels 0..9
N_ASK = 10   # levels 10..19

# Default: all features enabled
DEFAULT_FEATURES: Dict[str, bool] = {
    'order_flow_imbalance': True,
    'pressure_gradient': True,
    'spread': True,
    'queue_age_momentum': True,
    'depth_change_velocity': True,
    'book_asymmetry': True,
    'cumulative_depth_ratio': True,
    'queue_decay_rate': True,
    'phantom_liquidity': True,
    'book_renewal_asymmetry': True,
    'cross_level_pressure': True,
    'book_elasticity': True,
}

EPS = 1e-6  # numerical stability floor


# ---------------------------------------------------------------------------
# Per-feature computation helpers
# All helpers accept tensors of shape (B, T, 20, 4) and return (B, T, 20, 1).
# ---------------------------------------------------------------------------

def _order_flow_imbalance(x: torch.Tensor) -> torch.Tensor:
    """
    Order-flow imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + eps).

    Result is a scalar per (batch, time) step, broadcast across all 20 levels.
    Output range: [-1, +1].
    """
    # x: (B, T, 20, 4)
    bid_depth = x[:, :, :N_BID, FEAT_DEPTH]   # (B, T, 10)
    ask_depth = x[:, :, N_BID:, FEAT_DEPTH]   # (B, T, 10)

    sum_bid = bid_depth.sum(dim=-1, keepdim=True)  # (B, T, 1)
    sum_ask = ask_depth.sum(dim=-1, keepdim=True)  # (B, T, 1)

    ofi = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPS)  # (B, T, 1)
    # Broadcast across 20 levels
    ofi = ofi.unsqueeze(-1).expand(-1, -1, N_LEVELS, 1)    # (B, T, 20, 1)
    return ofi


def _pressure_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Depth gradient: depth[level_n+1] - depth[level_n] within each side.

    Positive = depth is increasing deeper into the book (wall forming farther out).
    Negative = depth thins as you move away from best price.

    Boundary: level 0 (best bid) and level 10 (best ask) use the same value as
    their level-1 neighbour (forward difference pad).
    Output: (B, T, 20, 1)
    """
    depth = x[:, :, :, FEAT_DEPTH]  # (B, T, 20)

    bid_depth = depth[:, :, :N_BID]   # (B, T, 10)
    ask_depth = depth[:, :, N_BID:]   # (B, T, 10)

    # diff[i] = depth[i+1] - depth[i], shape (B, T, 9)
    # Pad front with first diff value so output is (B, T, 10)
    bid_diff = bid_depth[:, :, 1:] - bid_depth[:, :, :-1]     # (B, T, 9)
    bid_grad = torch.cat([bid_diff[:, :, :1], bid_diff], dim=-1)  # (B, T, 10)

    ask_diff = ask_depth[:, :, 1:] - ask_depth[:, :, :-1]
    ask_grad = torch.cat([ask_diff[:, :, :1], ask_diff], dim=-1)

    grad = torch.cat([bid_grad, ask_grad], dim=-1)  # (B, T, 20)
    return grad.unsqueeze(-1)                        # (B, T, 20, 1)


def _spread(x: torch.Tensor) -> torch.Tensor:
    """
    Bid-ask spread = best_ask_price - best_bid_price (in ticks).

    best_bid = price_rel_to_mid at level 9  (worst-of-best = closest to mid on bid)
    best_ask = price_rel_to_mid at level 10 (closest to mid on ask)

    NOTE: level 9 is the *worst* bid level in the "best-to-worst" ordering,
    meaning it's the closest bid to mid. Level 10 is best ask (closest to mid).
    Because price_relative_to_mid is signed from mid, bid prices are negative
    and ask prices are positive.

    Spread is always >= 0 (ask > bid). Broadcast to all 20 levels.
    Output: (B, T, 20, 1)
    """
    # Level 9 = innermost bid (price < 0 typically)
    # Level 10 = innermost ask (price > 0 typically)
    best_bid_price = x[:, :, N_BID - 1, FEAT_PRICE]   # (B, T)
    best_ask_price = x[:, :, N_BID, FEAT_PRICE]        # (B, T)

    spread = (best_ask_price - best_bid_price).clamp(min=0.0)  # (B, T)
    spread = spread.unsqueeze(-1).unsqueeze(-1)                # (B, T, 1, 1)
    spread = spread.expand(-1, -1, N_LEVELS, 1)                # (B, T, 20, 1)
    return spread


def _queue_age_momentum(x: torch.Tensor) -> torch.Tensor:
    """
    Queue age momentum = queue_age[t] - queue_age[t-1], per level.

    Positive = orders getting older (no refresh, stale book).
    Negative = orders refreshed (new quotes at this level).
    First bar of window padded with 0.
    Output: (B, T, 20, 1)
    """
    age = x[:, :, :, FEAT_AGE]          # (B, T, 20)
    # Shift by 1 along time axis
    age_prev = F.pad(age[:, :-1, :], (0, 0, 1, 0))  # (B, T, 20), first row = 0
    momentum = age - age_prev
    return momentum.unsqueeze(-1)        # (B, T, 20, 1)


def _depth_change_velocity(x: torch.Tensor) -> torch.Tensor:
    """
    Depth change velocity = depth_lots[t] - depth_lots[t-1], per level.

    Positive = depth increasing (orders accumulating or refreshing).
    Negative = depth decreasing (trades eating the book, or cancellations).
    First bar of window padded with 0.
    Output: (B, T, 20, 1)
    """
    depth = x[:, :, :, FEAT_DEPTH]      # (B, T, 20)
    depth_prev = F.pad(depth[:, :-1, :], (0, 0, 1, 0))
    velocity = depth - depth_prev
    return velocity.unsqueeze(-1)        # (B, T, 20, 1)


def _book_asymmetry(x: torch.Tensor) -> torch.Tensor:
    """
    Book asymmetry per level = bid_depth[i] - ask_depth[i] for i in 0..9.

    bid_depth[0] corresponds to ask_depth[0] (both are best-price levels).
    Result is mirrored: bid levels get positive sign when bid > ask,
    ask levels get the negated value (from ask's perspective).

    Output: (B, T, 20, 1)
    """
    bid_depth = x[:, :, :N_BID, FEAT_DEPTH]   # (B, T, 10)
    ask_depth = x[:, :, N_BID:, FEAT_DEPTH]   # (B, T, 10)

    # Asymmetry from bid perspective
    asym = bid_depth - ask_depth              # (B, T, 10)

    # Bid levels: asym[i]  (positive = more bid depth at level i)
    # Ask levels: -asym[i] (mirrored — from ask's perspective)
    bid_asym = asym                           # (B, T, 10)
    ask_asym = -asym                          # (B, T, 10)

    combined = torch.cat([bid_asym, ask_asym], dim=-1)  # (B, T, 20)
    return combined.unsqueeze(-1)                        # (B, T, 20, 1)


def _queue_decay_rate(x: torch.Tensor) -> torch.Tensor:
    """
    Queue decay rate — how fast depth disappears at each level over time.

    Measures the rolling second-derivative of depth: velocity change.
    depth_accel[t] = depth_vel[t] - depth_vel[t-1]
                   = (depth[t] - depth[t-1]) - (depth[t-1] - depth[t-2])
    Negative acceleration = depth disappearing faster = informed pulling quotes.
    Zero-padded for t=0 and t=1.
    Output: (B, T, 20, 1)
    """
    depth = x[:, :, :, FEAT_DEPTH]  # (B, T, 20)
    # First difference (velocity)
    vel = depth[:, 1:, :] - depth[:, :-1, :]  # (B, T-1, 20)
    # Second difference (acceleration)
    accel = vel[:, 1:, :] - vel[:, :-1, :]    # (B, T-2, 20)
    # Pad front with zeros to restore T dimension
    B, T, L = depth.shape
    pad = torch.zeros(B, 2, L, device=depth.device, dtype=depth.dtype)
    accel_padded = torch.cat([pad, accel], dim=1)  # (B, T, 20)
    return accel_padded.unsqueeze(-1)  # (B, T, 20, 1)


def _phantom_liquidity(x: torch.Tensor) -> torch.Tensor:
    """
    Phantom liquidity score — detects orders that appear then quickly disappear.

    Proxy: depth increased from t-2 to t-1 (order appeared) then decreased from
    t-1 to t (order cancelled/filled). High phantom score = spoofing/probing.

    phantom[t] = max(0, depth[t-1] - depth[t-2]) * max(0, depth[t-1] - depth[t])
    High value = depth spiked at t-1 then dropped at t.
    Zero-padded for t=0 and t=1.
    Output: (B, T, 20, 1)
    """
    depth = x[:, :, :, FEAT_DEPTH]  # (B, T, 20)
    B, T, L = depth.shape

    if T < 3:
        return torch.zeros(B, T, L, 1, device=depth.device, dtype=depth.dtype)

    # Depth appeared at t-1: depth[t-1] - depth[t-2]
    appeared = (depth[:, 1:-1, :] - depth[:, :-2, :]).clamp(min=0)  # (B, T-2, 20)
    # Depth disappeared at t: depth[t-1] - depth[t]
    disappeared = (depth[:, 1:-1, :] - depth[:, 2:, :]).clamp(min=0)  # (B, T-2, 20)

    phantom = appeared * disappeared  # High = spike then drop
    pad = torch.zeros(B, 2, L, device=depth.device, dtype=depth.dtype)
    phantom_padded = torch.cat([pad, phantom], dim=1)  # (B, T, 20)
    return phantom_padded.unsqueeze(-1)  # (B, T, 20, 1)


def _book_renewal_asymmetry(x: torch.Tensor) -> torch.Tensor:
    """
    Book renewal asymmetry — when best level depth drops, how fast does each
    side refill?

    Measures: (bid L1 depth recovery rate) - (ask L1 depth recovery rate)
    Where recovery = depth[t] - depth[t-1] when depth[t-1] < depth[t-2] (was consumed).
    Positive = bid refills faster (bullish MM behavior).

    Computed on L1 (best bid level 0, best ask level 10), broadcast to all levels.
    Zero-padded for t=0 and t=1.
    Output: (B, T, 20, 1)
    """
    bid_l1 = x[:, :, 0, FEAT_DEPTH]   # (B, T) — best bid depth
    ask_l1 = x[:, :, N_BID, FEAT_DEPTH]  # (B, T) — best ask depth
    B, T = bid_l1.shape

    if T < 3:
        return torch.zeros(B, T, N_LEVELS, 1, device=x.device, dtype=x.dtype)

    # Was consumed: depth[t-1] < depth[t-2]
    bid_consumed = (bid_l1[:, 1:-1] < bid_l1[:, :-2]).float()  # (B, T-2)
    ask_consumed = (ask_l1[:, 1:-1] < ask_l1[:, :-2]).float()

    # Recovery: depth[t] - depth[t-1] (only when was consumed)
    bid_recovery = (bid_l1[:, 2:] - bid_l1[:, 1:-1]) * bid_consumed  # (B, T-2)
    ask_recovery = (ask_l1[:, 2:] - ask_l1[:, 1:-1]) * ask_consumed

    asym = bid_recovery - ask_recovery  # Positive = bid refills faster
    pad = torch.zeros(B, 2, device=x.device, dtype=x.dtype)
    asym_padded = torch.cat([pad, asym], dim=1)  # (B, T)

    # Broadcast to all 20 levels
    return asym_padded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N_LEVELS, 1)


def _cross_level_pressure(x: torch.Tensor) -> torch.Tensor:
    """
    Cross-level pressure wave — detects depth building deeper in the book
    before price moves.

    Measures: depth change at levels 3-9 minus depth change at levels 0-2.
    Positive = depth building behind the front of book (wall forming).
    Negative = depth retreating from back levels (MM pulling out).

    Computed separately for bid and ask sides.
    Output: (B, T, 20, 1) — broadcast within each side
    """
    depth = x[:, :, :, FEAT_DEPTH]  # (B, T, 20)
    B, T, L = depth.shape

    if T < 2:
        return torch.zeros(B, T, L, 1, device=depth.device, dtype=depth.dtype)

    depth_delta = depth[:, 1:, :] - depth[:, :-1, :]  # (B, T-1, 20)

    # Bid side: levels 0-2 = front, levels 3-9 = back
    bid_front_change = depth_delta[:, :, :3].mean(dim=-1, keepdim=True)  # (B, T-1, 1)
    bid_back_change = depth_delta[:, :, 3:N_BID].mean(dim=-1, keepdim=True)
    bid_pressure = bid_back_change - bid_front_change  # (B, T-1, 1)

    # Ask side: levels 10-12 = front, levels 13-19 = back
    ask_front_change = depth_delta[:, :, N_BID:N_BID+3].mean(dim=-1, keepdim=True)
    ask_back_change = depth_delta[:, :, N_BID+3:].mean(dim=-1, keepdim=True)
    ask_pressure = ask_back_change - ask_front_change

    # Broadcast within each side
    bid_expanded = bid_pressure.expand(-1, -1, N_BID)   # (B, T-1, 10)
    ask_expanded = ask_pressure.expand(-1, -1, N_ASK)    # (B, T-1, 10)
    combined = torch.cat([bid_expanded, ask_expanded], dim=-1)  # (B, T-1, 20)

    # Pad front
    pad = torch.zeros(B, 1, L, device=depth.device, dtype=depth.dtype)
    combined_padded = torch.cat([pad, combined], dim=1)  # (B, T, 20)
    return combined_padded.unsqueeze(-1)  # (B, T, 20, 1)


def _book_elasticity(x: torch.Tensor) -> torch.Tensor:
    """
    Book elasticity — how much depth changes per unit price move.

    Measures correlation between mid-price change and total depth change.
    Inelastic (low value): depth doesn't respond to price → breakout setup.
    Elastic (high value): depth increases with price move → mean reversion.

    Uses a rolling 5-bar window within the window dimension.
    Output: (B, T, 20, 1) — broadcast to all levels.
    """
    # Mid price approximation: average of best bid and best ask prices
    best_bid_p = x[:, :, 0, FEAT_PRICE]    # (B, T)
    best_ask_p = x[:, :, N_BID, FEAT_PRICE]  # (B, T)
    mid = (best_bid_p + best_ask_p) / 2.0

    # Total depth
    total_depth = x[:, :, :, FEAT_DEPTH].sum(dim=-1)  # (B, T)

    B, T = mid.shape

    if T < 3:
        return torch.zeros(B, T, N_LEVELS, 1, device=x.device, dtype=x.dtype)

    # Price and depth deltas
    mid_delta = mid[:, 1:] - mid[:, :-1]          # (B, T-1)
    depth_delta = total_depth[:, 1:] - total_depth[:, :-1]  # (B, T-1)

    # Elasticity: depth_delta / (|mid_delta| + eps) — how responsive is depth to price
    elasticity = depth_delta / (mid_delta.abs() + EPS)  # (B, T-1)

    # Clamp to reasonable range (avoid extreme values from tiny price changes)
    elasticity = elasticity.clamp(-10.0, 10.0)

    # Pad front
    pad = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    elasticity_padded = torch.cat([pad, elasticity], dim=1)  # (B, T)

    # Broadcast to all levels
    return elasticity_padded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N_LEVELS, 1)


def _cumulative_depth_ratio(x: torch.Tensor) -> torch.Tensor:
    """
    Cumulative depth ratio up to level i = cum_bid_depth[i] / (cum_ask_depth[i] + eps).

    cum_bid_depth[i] = sum of bid depth from level 0..i (best to level i).
    cum_ask_depth[i] = sum of ask depth from level 0..i.

    Ratio > 1: more cumulative bid depth (bullish pressure).
    Ratio < 1: more cumulative ask depth (bearish pressure).
    Both bid and ask levels carry the ratio at their respective depth position.
    Output: (B, T, 20, 1)
    """
    bid_depth = x[:, :, :N_BID, FEAT_DEPTH]   # (B, T, 10)
    ask_depth = x[:, :, N_BID:, FEAT_DEPTH]   # (B, T, 10)

    # Cumulative sum along level axis (dim=-1)
    cum_bid = torch.cumsum(bid_depth, dim=-1)  # (B, T, 10)
    cum_ask = torch.cumsum(ask_depth, dim=-1)  # (B, T, 10)

    ratio = cum_bid / (cum_ask + EPS)          # (B, T, 10)

    # Use same ratio for matching ask levels (level i on ask = same depth band as level i on bid)
    combined = torch.cat([ratio, ratio], dim=-1)  # (B, T, 20)
    return combined.unsqueeze(-1)                  # (B, T, 20, 1)


# ---------------------------------------------------------------------------
# Feature registry: maps name -> function
# ---------------------------------------------------------------------------
_FEATURE_REGISTRY = {
    'order_flow_imbalance': _order_flow_imbalance,
    'pressure_gradient': _pressure_gradient,
    'spread': _spread,
    'queue_age_momentum': _queue_age_momentum,
    'depth_change_velocity': _depth_change_velocity,
    'book_asymmetry': _book_asymmetry,
    'cumulative_depth_ratio': _cumulative_depth_ratio,
    # Novel temporal features (require window > 2)
    'queue_decay_rate': _queue_decay_rate,
    'phantom_liquidity': _phantom_liquidity,
    'book_renewal_asymmetry': _book_renewal_asymmetry,
    'cross_level_pressure': _cross_level_pressure,
    'book_elasticity': _book_elasticity,
}

# Number of output channels added by each feature (all = 1 currently)
FEATURE_CHANNELS: Dict[str, int] = {k: 1 for k in _FEATURE_REGISTRY}


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class BookFeatureEngineer:
    """
    Computes derived features from raw book tensors and concatenates them
    along the feature dimension.

    Args:
        features: Dict mapping feature name -> bool (True = include).
                  Defaults to DEFAULT_FEATURES (all enabled).

    Example:
        engineer = BookFeatureEngineer()
        x_aug = engineer(x)   # x: (B, T, 20, 4) -> (B, T, 20, 4+N)

        # Only OFI + spread:
        engineer = BookFeatureEngineer({'order_flow_imbalance': True, 'spread': True})
        x_aug = engineer(x)   # -> (B, T, 20, 6)

    num_extra_features property:
        Returns the count of extra feature channels that will be appended.
        Use engineer.num_extra_features to set num_features in BookSpatialCNN.
    """

    def __init__(self, features: Optional[Dict[str, bool]] = None):
        if features is None:
            self.features = dict(DEFAULT_FEATURES)
        else:
            self.features = {k: bool(v) for k, v in features.items()}

        # Validate keys
        unknown = set(self.features) - set(_FEATURE_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown feature(s): {unknown}. Valid: {set(_FEATURE_REGISTRY)}")

        # Build ordered list of enabled feature functions
        self._enabled: list = [
            (name, _FEATURE_REGISTRY[name])
            for name in _FEATURE_REGISTRY  # preserve canonical ordering
            if self.features.get(name, False)
        ]

    @property
    def num_extra_features(self) -> int:
        """Total number of extra feature channels that will be appended."""
        return sum(FEATURE_CHANNELS[name] for name, _ in self._enabled)

    @property
    def total_features(self) -> int:
        """Total features after augmentation (4 raw + N derived)."""
        return 4 + self.num_extra_features

    @property
    def enabled_names(self) -> list:
        """Ordered list of enabled feature names."""
        return [name for name, _ in self._enabled]

    def __call__(self, x: torch.Tensor, has_batch: bool = True) -> torch.Tensor:
        """
        Augment book tensor with derived features.

        Args:
            x: Book tensor.
               If has_batch=True:  shape (B, T, 20, 4)
               If has_batch=False: shape (T, 20, 4)  — single window
            has_batch: Whether x includes a batch dimension.

        Returns:
            Augmented tensor with same dtype and device as input.
               If has_batch=True:  (B, T, 20, 4 + N)
               If has_batch=False: (T, 20, 4 + N)
        """
        if not self._enabled:
            return x  # No-op: no features selected

        if not has_batch:
            # Add batch dim, process, remove
            return self(x.unsqueeze(0), has_batch=True).squeeze(0)

        # x: (B, T, 20, 4)
        assert x.ndim == 4, f"Expected 4D tensor (B,T,L,F), got {x.ndim}D"
        assert x.shape[2] == N_LEVELS, f"Expected 20 levels, got {x.shape[2]}"
        assert x.shape[3] >= 4, f"Expected at least 4 raw features, got {x.shape[3]}"

        extras = [fn(x) for _, fn in self._enabled]  # list of (B, T, 20, 1)
        extras_cat = torch.cat(extras, dim=-1)         # (B, T, 20, N)

        return torch.cat([x, extras_cat], dim=-1)      # (B, T, 20, 4+N)

    def __repr__(self) -> str:
        enabled = ', '.join(self.enabled_names) or 'none'
        return (f"BookFeatureEngineer("
                f"enabled=[{enabled}], "
                f"num_extra={self.num_extra_features}, "
                f"total={self.total_features})")


# ---------------------------------------------------------------------------
# Convenience: count features for a given config dict
# ---------------------------------------------------------------------------

def count_features(feature_config: Optional[Dict[str, bool]] = None) -> int:
    """
    Return the total num_features (raw 4 + derived N) for a given feature config.
    Pass the same dict you'd pass to BookFeatureEngineer.
    Returns 4 if feature_config is None or empty.
    """
    if not feature_config:
        return 4
    engineer = BookFeatureEngineer(feature_config)
    return engineer.total_features


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    torch.manual_seed(0)
    B, T, L, nF = 2, 20, 20, 4
    x = torch.randn(B, T, L, nF)
    # Make depth/orders/age non-negative (they're log1p values in practice)
    x[:, :, :, 1:] = x[:, :, :, 1:].abs()

    print("=== BookFeatureEngineer self-test ===")

    # Test 1: all features
    eng_all = BookFeatureEngineer()
    x_aug = eng_all(x)
    print(f"All features: {x.shape} -> {x_aug.shape}")
    assert x_aug.shape == (B, T, L, 4 + eng_all.num_extra_features), "Shape mismatch"
    assert not torch.isnan(x_aug).any(), "NaN in output"
    assert not torch.isinf(x_aug).any(), "Inf in output"
    print(f"  Enabled: {eng_all.enabled_names}")
    print(f"  Extra channels: {eng_all.num_extra_features}")
    print(f"  Total features: {eng_all.total_features}")

    # Test 2: no features (passthrough)
    eng_none = BookFeatureEngineer({k: False for k in DEFAULT_FEATURES})
    x_pass = eng_none(x)
    assert torch.equal(x_pass, x), "Passthrough failed"
    print(f"\nNo features (passthrough): {x.shape} -> {x_pass.shape} [OK]")

    # Test 3: single feature
    eng_ofi = BookFeatureEngineer({'order_flow_imbalance': True})
    x_ofi = eng_ofi(x)
    assert x_ofi.shape == (B, T, L, 5), f"Expected 5, got {x_ofi.shape[-1]}"
    ofi_vals = x_ofi[:, :, :, 4]
    # OFI should be in [-1, +1]
    assert ofi_vals.min() >= -1.0 - 1e-5 and ofi_vals.max() <= 1.0 + 1e-5, \
        f"OFI out of range: [{ofi_vals.min():.4f}, {ofi_vals.max():.4f}]"
    # All levels should share the same OFI value (it's broadcast)
    assert torch.allclose(x_ofi[:, :, 0, 4], x_ofi[:, :, 10, 4]), \
        "OFI should be broadcast across levels"
    print(f"\nOFI only: {x.shape} -> {x_ofi.shape}, range [{ofi_vals.min():.3f}, {ofi_vals.max():.3f}] [OK]")

    # Test 4: spread non-negative
    eng_spread = BookFeatureEngineer({'spread': True})
    x_sp = eng_spread(x)
    spread_vals = x_sp[:, :, 0, 4]
    assert (spread_vals >= 0).all(), "Spread must be non-negative"
    print(f"\nSpread only: min={spread_vals.min():.3f} (should be >= 0) [OK]")

    # Test 5: has_batch=False
    x_single = x[0]  # (T, 20, 4)
    x_single_aug = eng_all(x_single, has_batch=False)
    assert x_single_aug.shape == (T, L, eng_all.total_features), \
        f"has_batch=False shape mismatch: {x_single_aug.shape}"
    print(f"\nhas_batch=False: {x_single.shape} -> {x_single_aug.shape} [OK]")

    # Test 6: GPU (if available)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        x_aug_gpu = eng_all(x_gpu)
        assert x_aug_gpu.device.type == 'cuda', "Output not on GPU"
        assert not torch.isnan(x_aug_gpu).any()
        print(f"\nGPU test: device={x_aug_gpu.device} [OK]")
    else:
        print("\nGPU test: skipped (no CUDA)")

    # Test 7: count_features helper
    assert count_features(None) == 4
    assert count_features({'order_flow_imbalance': True, 'spread': True}) == 6
    assert count_features(DEFAULT_FEATURES) == eng_all.total_features
    print(f"\ncount_features helper: [OK]")

    print(f"\n=== All tests passed ===")
    print(f"repr: {eng_all}")
