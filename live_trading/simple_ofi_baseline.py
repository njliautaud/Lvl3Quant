"""
Simple OFI Baseline - Monday Deploy
====================================
Clean, simple order flow imbalance signals.
Low risk, proven microstructure, deployable.

Rules:
- Long when OFI > threshold AND spread < max_spread
- Short when OFI < -threshold AND spread < max_spread
- Exit on spread widening or OFI reversal
- Hard stop: 2 ticks
- Target: 1.5 ticks

Risk-adjusted by design:
- Only trade tight spreads (low cost)
- Quick exits (low drawdown)
- Simple logic (robust, no overfit)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class OFISignal:
    """OFI-based signal"""
    direction: int  # 1=long, -1=short, 0=flat
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

class SimpleOFIStrategy:
    """
    Production-ready OFI baseline
    Sharpe target: 2.0+
    Max DD target: 10%
    """

    def __init__(self,
                 ofi_threshold: float = 50.0,    # OFI signal threshold
                 max_spread: float = 1.5,        # Only trade if spread < 1.5 ticks
                 stop_ticks: float = 2.0,        # Hard stop
                 target_ticks: float = 1.5,      # Profit target
                 lookback: int = 20):            # OFI lookback window
        self.ofi_threshold = ofi_threshold
        self.max_spread = max_spread
        self.stop_ticks = stop_ticks
        self.target_ticks = target_ticks
        self.lookback = lookback

    def compute_ofi(self, events: np.ndarray) -> float:
        """
        Order flow imbalance over lookback window
        events: (N, 6) array [time_delta, event_type, side, price, qty, spread]
        """
        # Get last N events
        recent = events[-self.lookback:] if len(events) > self.lookback else events

        # Filter to trades only (event_type==2)
        trades = recent[recent[:, 1] == 2]

        if len(trades) == 0:
            return 0.0

        # OFI = (buy_volume - sell_volume)
        # side: 0=bid, 1=ask
        buy_vol = trades[trades[:, 2] == 1, 4].sum()  # Ask-side trades (buys)
        sell_vol = trades[trades[:, 2] == 0, 4].sum()  # Bid-side trades (sells)

        return buy_vol - sell_vol

    def generate_signal(self,
                       events: np.ndarray,
                       current_bid: float,
                       current_ask: float) -> OFISignal:
        """Generate signal from current market state"""

        spread = current_ask - current_bid

        # Don't trade wide spreads (high cost)
        if spread > self.max_spread:
            return OFISignal(direction=0, confidence=0.0)

        # Compute OFI
        ofi = self.compute_ofi(events)

        # Signal logic
        if ofi > self.ofi_threshold:
            # Strong buying pressure -> go long
            mid = (current_bid + current_ask) / 2
            return OFISignal(
                direction=1,
                confidence=min(abs(ofi) / (self.ofi_threshold * 2), 1.0),
                entry_price=mid,
                stop_price=mid - self.stop_ticks * 0.25,  # ES tick size
                target_price=mid + self.target_ticks * 0.25
            )

        elif ofi < -self.ofi_threshold:
            # Strong selling pressure -> go short
            mid = (current_bid + current_ask) / 2
            return OFISignal(
                direction=-1,
                confidence=min(abs(ofi) / (self.ofi_threshold * 2), 1.0),
                entry_price=mid,
                stop_price=mid + self.stop_ticks * 0.25,
                target_price=mid - self.target_ticks * 0.25
            )

        else:
            # No signal
            return OFISignal(direction=0, confidence=0.0)

    def backtest(self, events_file: str) -> dict:
        """Simple backtest to validate before deploy"""
        # TODO: Load events, run through strategy, compute Sharpe/Sortino
        pass

# Deploy config
PRODUCTION_CONFIG = {
    'ofi_threshold': 50.0,
    'max_spread': 1.5,
    'stop_ticks': 2.0,
    'target_ticks': 1.5,
    'lookback': 20,
    'max_position': 1,  # Conservative size
}

if __name__ == "__main__":
    print("Simple OFI Baseline - Production Config:")
    for k, v in PRODUCTION_CONFIG.items():
        print(f"  {k}: {v}")
    print("\nTarget metrics:")
    print("  Sharpe: ≥2.0")
    print("  Sortino: ≥2.5")
    print("  Max DD: ≤10%")
    print("  Win rate: ~55%+")
    print("\nReady for Monday deploy pending backtest validation.")
