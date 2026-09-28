"""
Real-Time MBO Data Feed
========================

Provides unified interface for:
1. Live MBO data from Rithmic/Databento
2. Historical replay from .npz event files
3. Rate-controlled replay for testing

Output: Stream of (timestamp, event_type, side, price, qty, spread, time_delta) tuples
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MBOEvent:
    """Standardized MBO event format"""
    timestamp: float  # Unix timestamp (seconds)
    event_type: int   # 0=add, 1=cancel, 2=modify, 3=trade, 4=fill
    side: int         # 0=bid, 1=ask, 2=none
    price: float      # Price level
    qty: float        # Quantity
    spread: float     # Bid-ask spread (ticks)
    time_delta: float # Time since previous event (seconds)

    # Optional fields for diagnostics
    symbol: Optional[str] = None
    exchange_timestamp: Optional[float] = None


class DataFeedBase(ABC):
    """Base class for all data feeds"""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def subscribe(self, symbol: str, exchange: str = "CME") -> None:
        """Subscribe to symbol data"""
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[MBOEvent]:
        """Stream events asynchronously"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnect"""
        pass


class ReplayFeed(DataFeedBase):
    """Replay MBO events from preprocessed .npz files"""

    def __init__(
        self,
        data_dir: str,
        replay_speed: float = 1.0,  # 1.0 = real-time, 0 = unlimited
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.replay_speed = float(replay_speed)
        self.start_date = start_date
        self.end_date = end_date
        self._files: List[Path] = []
        self._connected = False

    async def connect(self) -> None:
        """Load available data files"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find all event files
        pattern = "*_mbo_events.npz"
        all_files = sorted(self.data_dir.glob(pattern))

        # Filter by date range
        if self.start_date or self.end_date:
            self._files = []
            for f in all_files:
                date_str = f.stem.split('_')[0]  # e.g., "20251201_mbo_events" -> "20251201"
                if self.start_date and date_str < self.start_date:
                    continue
                if self.end_date and date_str > self.end_date:
                    continue
                self._files.append(f)
        else:
            self._files = all_files

        if not self._files:
            raise ValueError(f"No data files found in {self.data_dir}")

        logger.info(f"ReplayFeed connected: {len(self._files)} files from {self._files[0].stem} to {self._files[-1].stem}")
        self._connected = True

    async def subscribe(self, symbol: str, exchange: str = "CME") -> None:
        """No-op for replay (symbol is implicit in data files)"""
        logger.info(f"ReplayFeed subscribed to {symbol} on {exchange} (replay mode)")

    async def stream(self) -> AsyncIterator[MBOEvent]:
        """Stream events from files"""
        if not self._connected:
            raise RuntimeError("Feed not connected. Call connect() first.")

        for file_path in self._files:
            logger.info(f"Replaying {file_path.name}...")

            try:
                data = np.load(str(file_path))
                events = data['events']  # (N, 6): [time_delta_log, event_type, side, price_rel_ticks, qty_log, spread_ticks]
                timestamps = data['timestamps']  # (N,) int64 nanoseconds

                # Convert from log/relative space back to absolute
                N = len(events)
                prev_ts = None

                for i in range(N):
                    # Parse features
                    time_delta_log = events[i, 0]
                    event_type = int(events[i, 1])
                    side = int(events[i, 2])
                    price_rel_ticks = events[i, 3]  # Relative to mid (ticks)
                    qty_log = events[i, 4]
                    spread_ticks = events[i, 5]

                    # Convert timestamp from nanoseconds to seconds
                    ts_sec = timestamps[i] / 1e9

                    # Time delta in seconds
                    if prev_ts is None:
                        time_delta = 0.0
                    else:
                        time_delta = max(0.0, ts_sec - prev_ts)
                    prev_ts = ts_sec

                    # De-log quantities
                    qty = np.expm1(qty_log)  # inverse of log1p

                    # Price is relative to mid — we'll use a placeholder absolute price
                    # In real system, maintain running mid-price estimate
                    price = 5000.0 + price_rel_ticks * 0.25  # ES ~5000, tick=0.25

                    # Create event
                    event = MBOEvent(
                        timestamp=ts_sec,
                        event_type=event_type,
                        side=side,
                        price=price,
                        qty=qty,
                        spread=spread_ticks * 0.25,  # Convert ticks to price
                        time_delta=time_delta,
                        symbol="ES",
                        exchange_timestamp=ts_sec,
                    )

                    # Rate limiting for realistic replay
                    if self.replay_speed > 0:
                        await asyncio.sleep(time_delta / self.replay_speed)

                    yield event

                logger.info(f"Completed {file_path.name}: {N:,} events")

            except Exception as e:
                logger.error(f"Error replaying {file_path.name}: {e}")
                continue

    async def disconnect(self) -> None:
        """Cleanup"""
        logger.info("ReplayFeed disconnected")
        self._connected = False


class LiveFeed(DataFeedBase):
    """Live MBO feed from Rithmic (wrapper around existing rithmic_client.py)"""

    def __init__(self):
        # Import here to avoid circular dependency
        from live_trading_linux.rithmic_client import RithmicClient
        self.client = RithmicClient()
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._connected = False
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._last_ts: Optional[float] = None

    async def connect(self) -> None:
        """Connect to Rithmic"""
        await self.client.connect()
        self.client.set_md_callback(self._on_md)
        self._connected = True
        logger.info("LiveFeed connected to Rithmic")

    async def subscribe(self, symbol: str, exchange: str = "CME") -> None:
        """Subscribe to symbol"""
        await self.client.subscribe_md(symbol, exchange)
        logger.info(f"LiveFeed subscribed to {symbol} on {exchange}")

    async def _on_md(self, event) -> None:
        """Callback from Rithmic client"""
        from live_trading_linux.rithmic_client import BBOEvent, TradeEvent

        now = time.time()
        time_delta = 0.0 if self._last_ts is None else (now - self._last_ts)
        self._last_ts = now

        if isinstance(event, BBOEvent):
            # BBO update → synthesize add events
            if event.has_bid:
                self._best_bid = event.bid_price
                spread = self._best_ask - self._best_bid if self._best_ask > 0 else 0.0
                mbo_event = MBOEvent(
                    timestamp=now,
                    event_type=0,  # add
                    side=0,  # bid
                    price=event.bid_price,
                    qty=event.bid_size or 1,
                    spread=spread / 0.25,  # ticks
                    time_delta=time_delta,
                    symbol=event.symbol,
                    exchange_timestamp=event.ssboe + event.usecs * 1e-6,
                )
                await self._event_queue.put(mbo_event)

            if event.has_ask:
                self._best_ask = event.ask_price
                spread = self._best_ask - self._best_bid if self._best_bid > 0 else 0.0
                mbo_event = MBOEvent(
                    timestamp=now,
                    event_type=0,  # add
                    side=1,  # ask
                    price=event.ask_price,
                    qty=event.ask_size or 1,
                    spread=spread / 0.25,
                    time_delta=time_delta,
                    symbol=event.symbol,
                    exchange_timestamp=event.ssboe + event.usecs * 1e-6,
                )
                await self._event_queue.put(mbo_event)

        elif isinstance(event, TradeEvent):
            # Trade event
            spread = self._best_ask - self._best_bid if (self._best_bid > 0 and self._best_ask > 0) else 0.0
            side = 1 if event.aggressor == 1 else 0  # 1=buy, 2=sell
            mbo_event = MBOEvent(
                timestamp=now,
                event_type=3,  # trade
                side=side,
                price=event.trade_price,
                qty=event.trade_size,
                spread=spread / 0.25,
                time_delta=time_delta,
                symbol=event.symbol,
                exchange_timestamp=event.ssboe + event.usecs * 1e-6,
            )
            await self._event_queue.put(mbo_event)

    async def stream(self) -> AsyncIterator[MBOEvent]:
        """Stream events from queue"""
        if not self._connected:
            raise RuntimeError("Feed not connected")

        while True:
            event = await self._event_queue.get()
            yield event

    async def disconnect(self) -> None:
        """Disconnect from Rithmic"""
        await self.client.disconnect()
        self._connected = False
        logger.info("LiveFeed disconnected")


# Factory function
def create_feed(
    mode: str = "replay",
    data_dir: Optional[str] = None,
    replay_speed: float = 1.0,
    **kwargs
) -> DataFeedBase:
    """
    Create data feed based on mode.

    Args:
        mode: "replay" or "live"
        data_dir: Path to .npz files (replay mode only)
        replay_speed: Speed multiplier for replay (0 = unlimited)
        **kwargs: Additional mode-specific parameters

    Returns:
        DataFeedBase instance
    """
    if mode == "replay":
        if data_dir is None:
            data_dir = "/home/jupiter/Lvl3Quant/data/processed/mbo_events"
        return ReplayFeed(data_dir=data_dir, replay_speed=replay_speed, **kwargs)

    elif mode == "live":
        return LiveFeed()

    else:
        raise ValueError(f"Unknown feed mode: {mode}")
