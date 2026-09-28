"""
Bar-Based Feature Extraction for LGBM Models
==============================================

Converts streaming MBO events to 100ms bars and extracts features
matching the LGBM training pipeline.

Based on: /home/jupiter/Lvl3Quant/alpha_discovery/train_lgbm_sliding_60d.py
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bar:
    """Represents a single 100ms bar"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    event_count: int
    buy_volume: float
    sell_volume: float
    spread: float


class BarAggregator:
    """
    Aggregates streaming MBO events into 100ms bars

    Usage:
        aggregator = BarAggregator(bar_ms=100)

        for event in mbo_stream:
            bar = aggregator.add_event(event)
            if bar is not None:
                # Bar completed, extract features
                features = extractor.extract(bar)
    """

    def __init__(self, bar_ms: int = 100):
        self.bar_ms = bar_ms
        self.current_bar_start = None
        self.current_bar_events = []
        self.completed_bars = []
        self.max_bars = 50  # Keep last 50 bars for feature extraction

    def add_event(self, event) -> Optional[Bar]:
        """
        Add MBO event to aggregator

        Args:
            event: MBOEvent with fields:
                - timestamp (float): Event timestamp in seconds
                - event_type (int): 0=add, 1=cancel, 2=trade
                - side (int): 0=bid, 1=ask
                - price (float): Price level
                - qty (int): Quantity
                - spread (float): Bid-ask spread

        Returns:
            Bar if a bar was completed, None otherwise
        """
        timestamp_ms = event.timestamp * 1000

        # Initialize first bar
        if self.current_bar_start is None:
            self.current_bar_start = int(timestamp_ms / self.bar_ms) * self.bar_ms

        # Check if event belongs to current bar
        event_bar_start = int(timestamp_ms / self.bar_ms) * self.bar_ms

        if event_bar_start > self.current_bar_start:
            # Complete current bar
            completed = self._complete_bar()

            # Start new bar
            self.current_bar_start = event_bar_start
            self.current_bar_events = [event]

            return completed
        else:
            # Add to current bar
            self.current_bar_events.append(event)
            return None

    def _complete_bar(self) -> Optional[Bar]:
        """Complete and aggregate current bar"""
        if not self.current_bar_events:
            return None

        events = self.current_bar_events

        # Extract prices
        prices = [e.price for e in events]
        open_price = prices[0]
        close_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)

        # Volume aggregation
        total_volume = sum(e.qty for e in events)
        buy_volume = sum(e.qty for e in events if e.event_type == 2 and e.side == 0)
        sell_volume = sum(e.qty for e in events if e.event_type == 2 and e.side == 1)

        # Spread (average over bar)
        mean_spread = np.mean([e.spread for e in events])

        bar = Bar(
            timestamp=self.current_bar_start / 1000.0,  # Convert back to seconds
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            event_count=len(events),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            spread=mean_spread
        )

        # Store in history
        self.completed_bars.append(bar)
        if len(self.completed_bars) > self.max_bars:
            self.completed_bars.pop(0)

        return bar

    def get_recent_bars(self, lookback: int = 30) -> List[Bar]:
        """Get recent N bars for feature extraction"""
        return self.completed_bars[-lookback:] if len(self.completed_bars) >= lookback else []


class LGBMFeatureExtractor:
    """
    Extract LGBM features from bar window

    Matches feature extraction from train_lgbm_sliding_60d.py
    """

    def __init__(self, lookback: int = 30):
        """
        Args:
            lookback: Number of bars to use for feature extraction (default 30 = 3 seconds)
        """
        self.lookback = lookback

    def extract(self, bars: List[Bar]) -> Optional[np.ndarray]:
        """
        Extract 13 features from bar window

        Args:
            bars: List of Bar objects (length >= lookback)

        Returns:
            Feature vector (13 features) or None if insufficient bars
        """
        if len(bars) < self.lookback:
            return None

        # Use last 'lookback' bars
        window = bars[-self.lookback:]

        # Extract OHLCV arrays
        closes = np.array([b.close for b in window])
        volumes = np.array([b.volume for b in window])
        buy_vols = np.array([b.buy_volume for b in window])
        sell_vols = np.array([b.sell_volume for b in window])
        spreads = np.array([b.spread for b in window])
        event_counts = np.array([b.event_count for b in window])
        highs = np.array([b.high for b in window])
        lows = np.array([b.low for b in window])

        # Feature 1-5: Returns
        ret5 = (closes[-1] - closes[-5]) / (closes[-5] + 1e-8) if len(closes) >= 5 else 0.0
        ret10 = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8) if len(closes) >= 10 else 0.0
        ret_all = (closes[-1] - closes[0]) / (closes[0] + 1e-8) if closes[0] > 0 else 0.0

        rets = np.diff(closes) / (closes[:-1] + 1e-8) if len(closes) > 1 else np.array([0.0])
        ret_mean = float(rets.mean()) if len(rets) > 0 else 0.0
        ret_std = float(rets.std()) if len(rets) > 1 else 0.0

        # Feature 6-7: Volume
        vol_mean = float(volumes.mean())
        vol_ratio = float(volumes[-1] / (vol_mean + 1e-8))

        # Feature 8: Order flow imbalance
        total_trade_vol = buy_vols.sum() + sell_vols.sum()
        buy_imbal = float((buy_vols.sum() - sell_vols.sum()) / (total_trade_vol + 1e-8))

        # Feature 9-10: Spread
        sprd_mean = float(spreads.mean())
        sprd_dev = float(spreads[-1] - sprd_mean)

        # Feature 11: Event density
        n_mean = float(event_counts.mean())

        # Feature 12-13: Price range
        full_range = float(highs.max() - lows.min())
        pos_in_range = float(closes[-1] - lows.min()) / (full_range + 1e-8)

        # Assemble feature vector (must match training order)
        features = np.array([
            ret5, ret10, ret_all, ret_mean, ret_std,
            vol_mean, vol_ratio, buy_imbal, sprd_mean, sprd_dev,
            n_mean, full_range, pos_in_range
        ], dtype=np.float32)

        return features

    @property
    def feature_names(self) -> List[str]:
        """Return feature names in order"""
        return [
            'ret5', 'ret10', 'ret_all', 'ret_mean', 'ret_std',
            'vol_mean', 'vol_ratio', 'buy_imbal', 'sprd_mean', 'sprd_dev',
            'n_mean', 'full_range', 'pos_in_range'
        ]


# Example usage
if __name__ == "__main__":
    # Test with mock data
    from dataclasses import dataclass as dc

    @dc
    class MockEvent:
        timestamp: float
        event_type: int
        side: int
        price: float
        qty: int
        spread: float

    # Create aggregator and extractor
    aggregator = BarAggregator(bar_ms=100)
    extractor = LGBMFeatureExtractor(lookback=30)

    # Simulate events
    base_time = 1000.0
    base_price = 5000.0

    print("Simulating MBO event stream...")
    for i in range(1000):
        event = MockEvent(
            timestamp=base_time + i * 0.01,  # 10ms between events
            event_type=2 if i % 3 == 0 else 0,  # Every 3rd is a trade
            side=0 if i % 2 == 0 else 1,  # Alternate sides
            price=base_price + np.random.randn() * 0.25,
            qty=1 + int(abs(np.random.randn()) * 10),
            spread=0.25 + abs(np.random.randn()) * 0.1
        )

        bar = aggregator.add_event(event)
        if bar is not None:
            # Extract features when bar completes
            features = extractor.extract(aggregator.get_recent_bars())
            if features is not None:
                print(f"Bar @ {bar.timestamp:.1f}s: OHLC={bar.open:.2f}/{bar.high:.2f}/{bar.low:.2f}/{bar.close:.2f}, Features={features[:3]}")

    print(f"\n✓ Completed {len(aggregator.completed_bars)} bars")
    print(f"✓ Feature vector size: {len(extractor.feature_names)}")
    print(f"✓ Feature names: {', '.join(extractor.feature_names[:5])}...")
