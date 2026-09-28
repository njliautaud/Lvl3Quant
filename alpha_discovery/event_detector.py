"""
Event Detector — Discrete microstructure event detection from MBO features.

Detects 6 types of discrete market microstructure events from the existing
149-feature matrix. These events capture non-linear, burst-like patterns
that rolling averages miss:

1. SweepDetector: Aggressive price sweep through multiple levels
2. IcebergDetector: Hidden liquidity being executed without price movement
3. CancelStormDetector: Sudden burst of cancellations (spoofing/repositioning)
4. AbsorptionDetector: Passive side absorbing aggressive volume without moving price
5. SpoofDetector: Layering with high cancel asymmetry + fleeting orders
6. RegimeBreakDetector: Volatility regime transitions

Output: 14 event feature columns appended to the feature matrix.

All detectors operate on EXISTING features — no cache rebuild needed.
All thresholds are computed from rolling windows to avoid lookahead.
"""

import numpy as np
from typing import Dict, List, Tuple

from alpha_discovery.mbo_features import (
    get_feature_names,
    _rolling_mean,
    _rolling_std,
    _rolling_sum,
    _rolling_max,
)


# ============================================================================
# EVENT FEATURE NAMES (14 total)
# ============================================================================

EVENT_FEATURE_NAMES = [
    'sweep_event_buy',          # Buy-side sweep detected
    'sweep_event_sell',         # Sell-side sweep detected
    'iceberg_event_buy',        # Hidden buy liquidity execution
    'iceberg_event_sell',       # Hidden sell liquidity execution
    'cancel_storm_bid',         # Cancel storm on bid side
    'cancel_storm_ask',         # Cancel storm on ask side
    'absorption_event_bid',     # Bid absorbing sell aggression
    'absorption_event_ask',     # Ask absorbing buy aggression
    'spoof_event_bid',          # Spoofing/layering on bid side
    'spoof_event_ask',          # Spoofing/layering on ask side
    'regime_break_up',          # Vol regime break upward
    'regime_break_down',        # Vol regime break downward
    'event_intensity',          # Rolling sum of all events
    'event_diversity',          # How many event types fired recently
]

N_EVENT_FEATURES = len(EVENT_FEATURE_NAMES)


def get_event_feature_names() -> List[str]:
    """Return ordered list of event feature names."""
    return EVENT_FEATURE_NAMES.copy()


# ============================================================================
# FEATURE NAME → INDEX MAPPING
# ============================================================================

def _build_feature_index(feature_names: List[str]) -> Dict[str, int]:
    """Build name→index mapping for fast lookup."""
    return {name: idx for idx, name in enumerate(feature_names)}


# ============================================================================
# INDIVIDUAL EVENT DETECTORS
# ============================================================================

class SweepDetector:
    """
    Detects aggressive price sweeps through multiple order book levels.

    Produces CONTINUOUS confidence scores (0.0-1.0) combining:
    - Trade size spike z-score above rolling threshold
    - Aggressive imbalance magnitude
    - Price movement relative to rolling norm

    Uses existing features: max_trade_size, aggressive_imbalance, mid_ret_1
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (sweep_buy, sweep_sell) continuous confidence arrays."""
        max_ts = features[:, fidx['max_trade_size']]
        aggr_imb = features[:, fidx['aggressive_imbalance']]
        mid_ret_1 = features[:, fidx['mid_ret_1']]

        # Z-score of trade size spike (how far above 2-sigma threshold)
        ts_mean = _rolling_mean(max_ts, 200)
        ts_std = _rolling_std(max_ts, 200)
        size_z = np.clip((max_ts - ts_mean) / np.maximum(ts_std, 1e-8) - 2.0, 0, 4) / 4.0

        # Aggression intensity: how far past 0.5 threshold (continuous)
        buy_aggr_z = np.clip((aggr_imb - 0.5) / 0.5, 0, 1)
        sell_aggr_z = np.clip((-aggr_imb - 0.5) / 0.5, 0, 1)

        # Price movement z-score relative to rolling norm
        abs_ret = np.abs(mid_ret_1)
        ret_mean = _rolling_mean(abs_ret, 500)
        price_z = np.clip(abs_ret / np.maximum(ret_mean, 1e-8) - 1.0, 0, 4) / 4.0

        # Geometric mean: only > 0 when all components are active
        sweep_buy = (size_z * buy_aggr_z * price_z) ** (1.0 / 3.0)
        sweep_sell = (size_z * sell_aggr_z * price_z) ** (1.0 / 3.0)

        return sweep_buy.astype(np.float32), sweep_sell.astype(np.float32)


class IcebergDetector:
    """
    Detects hidden liquidity execution (iceberg orders).

    Continuous score combining:
    - Volume surge magnitude relative to rolling average
    - Price stability (inverse of movement)
    - L1 concentration (visible replenishment)

    Uses: buy_volume, sell_volume, bid_L1_conc, ask_L1_conc, mid_ret_1
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (iceberg_buy, iceberg_sell) continuous confidence arrays."""
        buy_vol = features[:, fidx['buy_volume']]
        sell_vol = features[:, fidx['sell_volume']]
        mid_ret = features[:, fidx['mid_ret_1']]
        bid_conc = features[:, fidx['bid_L1_conc']]
        ask_conc = features[:, fidx['ask_L1_conc']]

        avg_buy = _rolling_mean(buy_vol, 50)
        avg_sell = _rolling_mean(sell_vol, 50)

        # Volume surge z-score (how far above 2x average)
        sell_surge = np.clip(sell_vol / np.maximum(avg_sell, 1.0) - 2.0, 0, 5) / 5.0
        buy_surge = np.clip(buy_vol / np.maximum(avg_buy, 1.0) - 2.0, 0, 5) / 5.0

        # Price stability: inverse of movement (1.0 = no move, 0.0 = big move)
        ret_std = _rolling_std(mid_ret, 100)
        stability = np.clip(1.0 - np.abs(mid_ret) / np.maximum(ret_std, 1e-8), 0, 1)

        # L1 concentration (continuous, higher = more concentrated)
        bid_conc_z = np.clip(bid_conc - 0.15, 0, 0.85) / 0.85
        ask_conc_z = np.clip(ask_conc - 0.15, 0, 0.85) / 0.85

        # Iceberg buy: sell aggression absorbed by hidden bid
        iceberg_buy = (sell_surge * stability * bid_conc_z) ** (1.0 / 3.0)
        # Iceberg sell: buy aggression absorbed by hidden ask
        iceberg_sell = (buy_surge * stability * ask_conc_z) ** (1.0 / 3.0)

        return iceberg_buy.astype(np.float32), iceberg_sell.astype(np.float32)


class CancelStormDetector:
    """
    Detects sudden bursts of cancellations.

    Continuous score combining:
    - Cancel rate spike magnitude (z-score above rolling baseline)
    - Directional asymmetry (which side is cancelling more)

    Uses: cancel_count, cancel_to_add, cancel_side_imbalance
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (cancel_storm_bid, cancel_storm_ask) continuous confidence arrays."""
        cancel_ct = features[:, fidx['cancel_count']]
        cancel_imb = features[:, fidx['cancel_side_imbalance']]

        # Cancel spike magnitude (how far above 3x average)
        avg_cancel = _rolling_mean(cancel_ct, 100)
        spike_z = np.clip(cancel_ct / np.maximum(avg_cancel, 1.0) - 3.0, 0, 5) / 5.0

        # Directional confidence (how imbalanced the cancels are)
        bid_dir = np.clip((cancel_imb - 0.15) / 0.85, 0, 1)
        ask_dir = np.clip((-cancel_imb - 0.15) / 0.85, 0, 1)

        cancel_storm_bid = (spike_z * bid_dir) ** 0.5
        cancel_storm_ask = (spike_z * ask_dir) ** 0.5

        return cancel_storm_bid.astype(np.float32), cancel_storm_ask.astype(np.float32)


class AbsorptionDetector:
    """
    Detects absorption: aggressive volume hitting one side but price doesn't move.

    Continuous score combining:
    - Aggression magnitude (how much flow is hitting)
    - Price stability (how little it moved despite the flow)

    Uses: aggressive_buy_count, aggressive_sell_count, mid_ret_1
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (absorption_bid, absorption_ask) continuous confidence arrays."""
        aggr_buy = features[:, fidx['aggressive_buy_count']]
        aggr_sell = features[:, fidx['aggressive_sell_count']]
        mid_ret = features[:, fidx['mid_ret_1']]

        avg_aggr_buy = _rolling_mean(aggr_buy, 100)
        avg_aggr_sell = _rolling_mean(aggr_sell, 100)

        # Aggression surge z-score
        sell_aggr_z = np.clip(aggr_sell / np.maximum(avg_aggr_sell, 1.0) - 2.0, 0, 5) / 5.0
        buy_aggr_z = np.clip(aggr_buy / np.maximum(avg_aggr_buy, 1.0) - 2.0, 0, 5) / 5.0

        # Price stability (inverse of movement magnitude)
        ret_std = _rolling_std(mid_ret, 100)
        stability = np.clip(1.0 - np.abs(mid_ret) / np.maximum(ret_std, 1e-8), 0, 1)

        # Absorption = high aggression × price stability
        absorption_bid = (sell_aggr_z * stability) ** 0.5
        absorption_ask = (buy_aggr_z * stability) ** 0.5

        return absorption_bid.astype(np.float32), absorption_ask.astype(np.float32)


class SpoofDetector:
    """
    Detects spoofing/layering: extreme cancel asymmetry + fleeting orders.

    Continuous score combining:
    - Cancel asymmetry z-score (how extreme the forward/backward ratio is)
    - Fleeting ratio z-score (how elevated the quick-cancel rate is)
    - Directional intensity (which side is being spoofed)

    Uses: cancel_asym_5, fleeting_ratio_5, cancel_side_imbalance
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (spoof_bid, spoof_ask) continuous confidence arrays."""
        cancel_asym = features[:, fidx['cancel_asym_5']]
        fleeting = features[:, fidx['fleeting_ratio_5']]
        cancel_imb = features[:, fidx['cancel_side_imbalance']]

        # Adaptive z-scores
        asym_mean = _rolling_mean(cancel_asym, 200)
        asym_std = _rolling_std(cancel_asym, 200)
        asym_z = np.clip(
            (np.abs(cancel_asym) - np.abs(asym_mean)) / np.maximum(asym_std, 1e-8) - 2.0,
            0, 4
        ) / 4.0

        fleeting_mean = _rolling_mean(fleeting, 200)
        fleeting_std = _rolling_std(fleeting, 200)
        fleeting_z = np.clip(
            (fleeting - fleeting_mean) / np.maximum(fleeting_std, 1e-8) - 1.5,
            0, 4
        ) / 4.0

        # Combined spoof confidence
        spoof_conf = (asym_z * fleeting_z) ** 0.5

        # Directional confidence
        bid_dir = np.clip((cancel_imb - 0.1) / 0.9, 0, 1)
        ask_dir = np.clip((-cancel_imb - 0.1) / 0.9, 0, 1)

        spoof_bid = spoof_conf * bid_dir
        spoof_ask = spoof_conf * ask_dir

        return spoof_bid.astype(np.float32), spoof_ask.astype(np.float32)


class RegimeBreakDetector:
    """
    Detects volatility regime transitions.

    Continuous score measuring magnitude of vol regime change,
    normalized by rolling volatility of the regime ratio itself.

    Uses: vol_regime, rvol_10, rvol_50
    """

    def detect(self, features: np.ndarray, fidx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (regime_break_up, regime_break_down) continuous confidence arrays."""
        vol_regime = features[:, fidx['vol_regime']]

        # Regime change magnitude (difference in vol ratio)
        regime_diff = np.diff(vol_regime, prepend=vol_regime[0])

        # Adaptive z-score: how many sigma is this change?
        diff_std = _rolling_std(regime_diff, 100)
        regime_z = regime_diff / np.maximum(diff_std, 1e-8)

        # Continuous confidence: clip at 2-sigma (regime break starts), scale to 0-1
        regime_break_up = np.clip((regime_z - 2.0) / 4.0, 0, 1).astype(np.float32)
        regime_break_down = np.clip((-regime_z - 2.0) / 4.0, 0, 1).astype(np.float32)

        return regime_break_up, regime_break_down


# ============================================================================
# ORCHESTRATION
# ============================================================================

class EventDetectionPipeline:
    """
    Runs all event detectors and produces the 14-column event feature matrix.

    Usage:
        pipeline = EventDetectionPipeline()
        event_features = pipeline.detect_all(features, feature_names)
        # event_features.shape = (N, 14)
    """

    def __init__(self):
        self.detectors = {
            'sweep': SweepDetector(),
            'iceberg': IcebergDetector(),
            'cancel_storm': CancelStormDetector(),
            'absorption': AbsorptionDetector(),
            'spoof': SpoofDetector(),
            'regime_break': RegimeBreakDetector(),
        }

    def detect_all(
        self,
        features: np.ndarray,
        feature_names: List[str],
        day_boundaries: List[int] = None,
    ) -> np.ndarray:
        """
        Run all detectors and return event feature matrix.

        Args:
            features: (N, F) feature matrix with F >= 149
            feature_names: list of feature names matching columns
            day_boundaries: optional day boundary indices for masking

        Returns:
            (N, 14) float32 event feature matrix
        """
        N = features.shape[0]
        fidx = _build_feature_index(feature_names)

        # Validate required features exist
        required = [
            'max_trade_size', 'aggressive_imbalance', 'mid_ret_1',
            'buy_volume', 'sell_volume', 'bid_L1_conc', 'ask_L1_conc',
            'cancel_count', 'cancel_side_imbalance',
            'aggressive_buy_count', 'aggressive_sell_count',
            'cancel_asym_5', 'fleeting_ratio_5',
            'vol_regime',
        ]
        missing = [f for f in required if f not in fidx]
        if missing:
            raise ValueError(f"Missing required features for event detection: {missing}")

        # Run each detector
        sweep_buy, sweep_sell = self.detectors['sweep'].detect(features, fidx)
        iceberg_buy, iceberg_sell = self.detectors['iceberg'].detect(features, fidx)
        cancel_bid, cancel_ask = self.detectors['cancel_storm'].detect(features, fidx)
        absorb_bid, absorb_ask = self.detectors['absorption'].detect(features, fidx)
        spoof_bid, spoof_ask = self.detectors['spoof'].detect(features, fidx)
        regime_up, regime_down = self.detectors['regime_break'].detect(features, fidx)

        # Summary features
        # Total event intensity: rolling mean of all confidence scores over 10 bars
        all_events = (
            sweep_buy + sweep_sell +
            iceberg_buy + iceberg_sell +
            cancel_bid + cancel_ask +
            absorb_bid + absorb_ask +
            spoof_bid + spoof_ask +
            regime_up + regime_down
        )
        event_intensity = _rolling_mean(all_events, 10)

        # Event diversity: rolling mean of per-type max confidence over 10 bars
        # Captures how many different event types are active AND their intensity
        type_intensities = np.column_stack([
            _rolling_mean(np.maximum(sweep_buy, sweep_sell), 10),
            _rolling_mean(np.maximum(iceberg_buy, iceberg_sell), 10),
            _rolling_mean(np.maximum(cancel_bid, cancel_ask), 10),
            _rolling_mean(np.maximum(absorb_bid, absorb_ask), 10),
            _rolling_mean(np.maximum(spoof_bid, spoof_ask), 10),
            _rolling_mean(np.maximum(regime_up, regime_down), 10),
        ]).astype(np.float32)
        # Entropy-inspired: sum of active intensities (continuous, higher = more diverse)
        event_diversity = type_intensities.sum(axis=1).astype(np.float32)

        # Assemble output: 14 columns in EVENT_FEATURE_NAMES order
        result = np.column_stack([
            sweep_buy,          # 0: sweep_event_buy
            sweep_sell,         # 1: sweep_event_sell
            iceberg_buy,        # 2: iceberg_event_buy
            iceberg_sell,       # 3: iceberg_event_sell
            cancel_bid,         # 4: cancel_storm_bid
            cancel_ask,         # 5: cancel_storm_ask
            absorb_bid,         # 6: absorption_event_bid
            absorb_ask,         # 7: absorption_event_ask
            spoof_bid,          # 8: spoof_event_bid
            spoof_ask,          # 9: spoof_event_ask
            regime_up,          # 10: regime_break_up
            regime_down,        # 11: regime_break_down
            event_intensity,    # 12: event_intensity
            event_diversity,    # 13: event_diversity
        ]).astype(np.float32)

        assert result.shape == (N, N_EVENT_FEATURES), \
            f"Expected ({N}, {N_EVENT_FEATURES}), got {result.shape}"

        # NaN-mask warmup bars at day boundaries (first 200 bars per day)
        # Event detectors use rolling windows up to 500 bars, but 200 is conservative
        WARMUP = 200
        if day_boundaries is not None and len(day_boundaries) > 1:
            for i in range(len(day_boundaries) - 1):
                day_start = day_boundaries[i]
                warmup_end = min(
                    day_start + WARMUP,
                    day_boundaries[i + 1] if i + 1 < len(day_boundaries) else N
                )
                result[day_start:warmup_end] = np.nan

        return result

    def get_event_stats(
        self,
        event_features: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each event type.

        Returns dict with fire rate, mean intensity, etc. for each event.
        """
        stats = {}
        for i, name in enumerate(EVENT_FEATURE_NAMES):
            col = event_features[:, i]
            valid = np.isfinite(col)
            if valid.sum() == 0:
                stats[name] = {'fire_rate': 0.0, 'mean': 0.0, 'n_valid': 0}
                continue

            vals = col[valid]
            stats[name] = {
                'fire_rate': float((vals > 0).mean()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'max': float(vals.max()),
                'n_fires': int((vals > 0).sum()),
                'n_valid': int(valid.sum()),
            }
        return stats


def compute_event_features(
    features: np.ndarray,
    feature_names: List[str],
    day_boundaries: List[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function: compute event features and return augmented matrix.

    Args:
        features: (N, F) original feature matrix
        feature_names: F feature names
        day_boundaries: optional day boundaries

    Returns:
        (augmented_features, augmented_names) where:
        - augmented_features: (N, F+14) with event features appended
        - augmented_names: F+14 feature names
    """
    pipeline = EventDetectionPipeline()
    event_feats = pipeline.detect_all(features, feature_names, day_boundaries)

    augmented = np.concatenate([features, event_feats], axis=1)
    augmented_names = list(feature_names) + get_event_feature_names()

    return augmented, augmented_names
