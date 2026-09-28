#!/usr/bin/env python3
"""
Fear + SmartMoney + Dip Cross-Type Confluence Paper Trading Engine
====================================================================

Signals (all 3 within 5-day window required):
1. VIX > 1.15 * VIX_60day_rolling_mean (market fear elevated)
2. Stock daily HL range < stock 60-day average HL range (spread narrowing = smart money)
3. Stock RSI(14) < 40 (stock dipping)

Universe: AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, UNH, LLY, AVGO, AMD
Position size: $300 per entry, max 2 concurrent positions
Exit rules:
- VIX drops below 60-day average (confluence broken)
- 21-day max hold reached
- +10% take profit
- -15% stop loss
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, asdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH", "LLY", "AVGO", "AMD"]
POSITION_SIZE = 300.0  # USD per entry
MAX_CONCURRENT_POSITIONS = 2

# Signal thresholds
VIX_MULTIPLIER = 1.15
RSI_THRESHOLD = 40
HL_RANGE_THRESHOLD = 1.0  # Current HL range must be < threshold * avg HL range

# Risk management
TP_PERCENT = 0.10  # +10%
SL_PERCENT = 0.15  # -15%
MAX_HOLD_DAYS = 21

# Directories
STATE_DIR = Path("/home/jupiter/Lvl3Quant/paper_engines/state")
LOG_DIR = Path("/home/jupiter/Lvl3Quant/paper_engines/logs")
STATE_FILE = STATE_DIR / "cross_type_confluence_state.json"
LOG_FILE = LOG_DIR / "cross_type_confluence_paper.log"

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_date: str  # YYYY-MM-DD
    entry_price: float
    shares: float
    entry_value: float  # Position size in USD
    signals_fired: Dict[str, bool]  # fear, smart_money, dip

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class ClosedTrade:
    """Represents a closed trade (realized P&L)."""
    symbol: str
    entry_date: str
    entry_price: float
    shares: float
    entry_value: float
    exit_date: str
    exit_price: float
    exit_value: float
    pnl: float  # Realized P&L in USD
    pnl_percent: float  # Realized P&L %
    exit_reason: str  # "TP", "SL", "TP_confluence", "hold_expired", "vix_dropped"
    signals_fired: Dict[str, bool]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class PaperTradingEngine:
    """Fear + SmartMoney + Dip cross-type confluence paper trading engine."""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.last_run_date: Optional[str] = None
        self.vix_signal_active: bool = False
        self.load_state()

    def load_state(self):
        """Load trading state from file."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)

                # Restore positions
                for sym, pos_dict in state.get("positions", {}).items():
                    self.positions[sym] = Position.from_dict(pos_dict)

                # Restore closed trades
                for trade_dict in state.get("closed_trades", []):
                    self.closed_trades.append(ClosedTrade.from_dict(trade_dict))

                self.last_run_date = state.get("last_run_date")
                self.vix_signal_active = state.get("vix_signal_active", False)

                logger.info(f"Loaded state: {len(self.positions)} open, {len(self.closed_trades)} closed")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        else:
            logger.info("No existing state file; starting fresh")

    def save_state(self):
        """Save trading state to file."""
        try:
            state = {
                "last_run_date": self.last_run_date,
                "vix_signal_active": self.vix_signal_active,
                "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
                "closed_trades": [t.to_dict() for t in self.closed_trades],
            }

            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved state: {len(self.positions)} open, {len(self.closed_trades)} closed")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def fetch_vix_data(self, lookback_days: int = 70) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Fetch VIX data and compute signal.

        Returns:
            (current_vix, vix_60d_mean, fear_signal_active)
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False)

            if vix_data.empty:
                logger.warning("No VIX data available")
                return None, None, False

            vix_close = float(vix_data["Close"].iloc[-1].item())
            vix_60d_mean = float(vix_data["Close"].tail(60).mean().item())

            fear_signal = vix_close > VIX_MULTIPLIER * vix_60d_mean

            logger.debug(f"VIX: {vix_close:.2f} | 60d mean: {vix_60d_mean:.2f} | Fear signal: {fear_signal}")

            return vix_close, vix_60d_mean, fear_signal

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return None, None, False

    def compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI(14) for a price series."""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    def check_signals(self, symbol: str, lookback_days: int = 70) -> Tuple[bool, bool, Dict]:
        """
        Check all three signals for a symbol.

        Returns:
            (smart_money_signal, dip_signal, data_dict)
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty or len(data) < 65:
                return False, False, {}

            # Signal 2: SmartMoney (HL range narrowing)
            hl_ranges = data["High"] - data["Low"]
            current_hl_range = float(hl_ranges.iloc[-1])
            avg_hl_range_60 = float(hl_ranges.tail(60).mean())
            smart_money_signal = current_hl_range < HL_RANGE_THRESHOLD * avg_hl_range_60

            # Signal 3: Dip (RSI < 40)
            closes = data["Close"].values
            rsi = self.compute_rsi(closes, period=14)
            dip_signal = rsi is not None and rsi < RSI_THRESHOLD

            data_dict = {
                "current_price": float(data["Close"].iloc[-1]),
                "current_hl_range": current_hl_range,
                "avg_hl_range_60": avg_hl_range_60,
                "rsi": float(rsi) if rsi is not None else None,
            }

            return smart_money_signal, dip_signal, data_dict

        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
            return False, False, {}

    def find_entry_signals(self, vix_signal_active: bool) -> List[Tuple[str, float, Dict]]:
        """
        Find all symbols with entry signals.

        Returns:
            List of (symbol, entry_price, signals_dict)
        """
        signals = []

        if not vix_signal_active:
            logger.debug("VIX signal inactive; no entries possible")
            return signals

        for symbol in UNIVERSE:
            smart_money_signal, dip_signal, data_dict = self.check_signals(symbol)

            # All 3 signals required
            if smart_money_signal and dip_signal:
                signals.append((
                    symbol,
                    data_dict["current_price"],
                    {
                        "fear": True,
                        "smart_money": smart_money_signal,
                        "dip": dip_signal,
                    }
                ))
                logger.info(f"{symbol}: ENTRY SIGNAL | Price: ${data_dict['current_price']:.2f} | RSI: {data_dict['rsi']:.1f}")

        return signals

    def check_exits(self, vix_close: float, vix_60d_mean: float) -> List[str]:
        """
        Check all open positions for exit conditions.

        Returns:
            List of symbols to close
        """
        to_close = []
        today = datetime.now().strftime("%Y-%m-%d")

        for symbol, pos in list(self.positions.items()):
            try:
                # Fetch current price
                current_data = yf.download(symbol, period="1d", progress=False)
                current_price = float(current_data["Close"].iloc[-1])
                current_value = current_price * pos.shares

                # Compute P&L
                unrealized_pnl = current_value - pos.entry_value
                unrealized_pnl_pct = unrealized_pnl / pos.entry_value

                exit_reason = None
                exit_price = None

                # Check exit conditions
                # 1. TP at +10%
                if unrealized_pnl_pct >= TP_PERCENT:
                    exit_reason = "TP"
                    exit_price = current_price
                    logger.info(f"{symbol}: TP HIT | Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f} | Gain: {unrealized_pnl_pct*100:.2f}%")

                # 2. SL at -15%
                elif unrealized_pnl_pct <= -SL_PERCENT:
                    exit_reason = "SL"
                    exit_price = current_price
                    logger.info(f"{symbol}: SL HIT | Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f} | Loss: {unrealized_pnl_pct*100:.2f}%")

                # 3. Max hold (21 days)
                entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                days_held = (datetime.strptime(today, "%Y-%m-%d") - entry_dt).days
                if days_held >= MAX_HOLD_DAYS:
                    exit_reason = "hold_expired"
                    exit_price = current_price
                    logger.info(f"{symbol}: MAX HOLD EXPIRED ({days_held} days) | Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f} | P&L: {unrealized_pnl_pct*100:.2f}%")

                # 4. VIX confluence broken (drops below 60d mean)
                elif vix_close is not None and vix_60d_mean is not None:
                    if vix_close < vix_60d_mean:
                        exit_reason = "vix_dropped"
                        exit_price = current_price
                        logger.info(f"{symbol}: CONFLUENCE BROKEN (VIX dropped) | Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f} | P&L: {unrealized_pnl_pct*100:.2f}%")

                # Record closed trade if exit triggered
                if exit_reason:
                    exit_value = exit_price * pos.shares
                    pnl = exit_value - pos.entry_value
                    pnl_pct = pnl / pos.entry_value

                    closed_trade = ClosedTrade(
                        symbol=symbol,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        shares=pos.shares,
                        entry_value=pos.entry_value,
                        exit_date=today,
                        exit_price=exit_price,
                        exit_value=exit_value,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        exit_reason=exit_reason,
                        signals_fired=pos.signals_fired,
                    )

                    self.closed_trades.append(closed_trade)
                    to_close.append(symbol)

            except Exception as e:
                logger.error(f"Error checking exits for {symbol}: {e}")

        return to_close

    def run_daily(self):
        """Run the daily trading logic."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Skip if already ran today
        if self.last_run_date == today:
            logger.info(f"Already ran today ({today}); skipping")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"DAILY RUN: {today}")
        logger.info(f"{'='*80}\n")

        # Step 1: Fetch VIX and evaluate fear signal
        vix_close, vix_60d_mean, vix_signal_active = self.fetch_vix_data()
        self.vix_signal_active = vix_signal_active

        if vix_close is None:
            logger.error("Cannot proceed without VIX data")
            return

        # Step 2: Check exits for open positions
        to_close = self.check_exits(vix_close, vix_60d_mean)
        for symbol in to_close:
            del self.positions[symbol]

        # Step 3: Check for new entries (only if we have room and VIX signal active)
        if len(self.positions) < MAX_CONCURRENT_POSITIONS:
            entry_signals = self.find_entry_signals(vix_signal_active)

            for symbol, entry_price, signals_dict in entry_signals:
                if symbol not in self.positions and len(self.positions) < MAX_CONCURRENT_POSITIONS:
                    shares = POSITION_SIZE / entry_price

                    self.positions[symbol] = Position(
                        symbol=symbol,
                        entry_date=today,
                        entry_price=entry_price,
                        shares=shares,
                        entry_value=POSITION_SIZE,
                        signals_fired=signals_dict,
                    )

                    logger.info(f"{symbol}: ENTRY | Price: ${entry_price:.2f} | Shares: {shares:.2f}")

        # Step 4: Print daily summary
        self._print_summary(today, vix_close, vix_60d_mean)

        # Step 5: Save state
        self.last_run_date = today
        self.save_state()

    def _print_summary(self, date: str, vix_close: float, vix_60d_mean: float):
        """Print daily trading summary."""
        summary_lines = [
            "\n" + "="*80,
            f"DAILY SUMMARY: {date}",
            "="*80,
            f"VIX: {vix_close:.2f} | 60d Avg: {vix_60d_mean:.2f} | Fear Signal: {'ACTIVE' if vix_close > 1.15 * vix_60d_mean else 'INACTIVE'}",
            "",
            f"Open Positions: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS}",
        ]

        # Open positions details
        if self.positions:
            total_unrealized = 0.0
            for symbol, pos in self.positions.items():
                try:
                    current_data = yf.download(symbol, period="1d", progress=False)
                    current_price = float(current_data["Close"].iloc[-1])
                    current_value = current_price * pos.shares
                    unrealized_pnl = current_value - pos.entry_value
                    unrealized_pnl_pct = (unrealized_pnl / pos.entry_value) * 100
                    total_unrealized += unrealized_pnl

                    entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                    today_dt = datetime.strptime(date, "%Y-%m-%d")
                    days_held = (today_dt - entry_dt).days

                    summary_lines.append(
                        f"  {symbol:6s} | Entry: ${pos.entry_price:7.2f} | Current: ${current_price:7.2f} | "
                        f"P&L: ${unrealized_pnl:8.2f} ({unrealized_pnl_pct:+6.2f}%) | Held: {days_held}d"
                    )
                except Exception as e:
                    summary_lines.append(f"  {symbol:6s} | Error fetching price: {e}")

            summary_lines.append(f"\nTotal Unrealized P&L: ${total_unrealized:+.2f}")
        else:
            summary_lines.append("  (None)")

        # Closed trades stats
        if self.closed_trades:
            realized_pnl = sum(t.pnl for t in self.closed_trades)
            win_count = sum(1 for t in self.closed_trades if t.pnl > 0)
            loss_count = sum(1 for t in self.closed_trades if t.pnl < 0)
            win_rate = (win_count / len(self.closed_trades) * 100) if self.closed_trades else 0

            summary_lines.extend([
                "",
                f"Closed Trades: {len(self.closed_trades)} total",
                f"  Wins: {win_count} | Losses: {loss_count} | Win Rate: {win_rate:.1f}%",
                f"  Total Realized P&L: ${realized_pnl:+.2f}",
            ])
        else:
            summary_lines.extend([
                "",
                "Closed Trades: 0",
            ])

        summary_lines.append("="*80 + "\n")

        summary_text = "\n".join(summary_lines)
        print(summary_text)
        logger.info(summary_text)

    def print_trade_history(self):
        """Print detailed closed trade history."""
        if not self.closed_trades:
            print("No closed trades yet.")
            return

        print("\n" + "="*120)
        print("CLOSED TRADE HISTORY")
        print("="*120)

        for trade in self.closed_trades:
            print(
                f"{trade.symbol:6s} | Entry: {trade.entry_date} @ ${trade.entry_price:7.2f} | "
                f"Exit: {trade.exit_date} @ ${trade.exit_price:7.2f} | "
                f"P&L: ${trade.pnl:8.2f} ({trade.pnl_percent*100:+6.2f}%) | "
                f"Reason: {trade.exit_reason:20s} | Shares: {trade.shares:.2f}"
            )

        print("="*120 + "\n")


def main():
    engine = PaperTradingEngine()
    engine.run_daily()


if __name__ == "__main__":
    main()
