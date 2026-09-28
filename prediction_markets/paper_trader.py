#!/usr/bin/env python3
"""
Polymarket Paper Trading Simulator.

Simulates trades using real Polymarket prices without placing real orders.
Uses edge_detector to find opportunities and Kelly criterion for position sizing.

Features:
  - Real-time price tracking via Polymarket API
  - Edge-based entry signals from edge_detector.py
  - Half-Kelly position sizing with 5% max per market
  - Stop-loss and take-profit automation
  - Full P&L tracking with win rate stats
  - Persistent state across sessions

Usage:
    python -m trading_agents.prediction_markets.paper_trader --run
    python -m trading_agents.prediction_markets.paper_trader --run --min-score 80
    python -m trading_agents.prediction_markets.paper_trader --status
    python -m trading_agents.prediction_markets.paper_trader --history
    python -m trading_agents.prediction_markets.paper_trader --reset
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("paper_trader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from polymarket_client import PolymarketScanner, TAKER_FEE, MAKER_FEE
from edge_detector import EdgeDetector, load_price_history

# Load .env
ENV_FILE = THIS_DIR.parents[1] / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v

# State files
PAPER_STATE_FILE = DATA_DIR / "paper_trades.json"
PAPER_LOG_FILE = DATA_DIR / "paper_trades_log.jsonl"

# Risk parameters
DEFAULT_BANKROLL = 10_000.0     # Simulated starting capital (USDC)
MAX_POSITION_PCT = 0.05         # 5% max per market
STOP_LOSS_PCT = 0.20            # 20% loss triggers stop
TAKE_PROFIT_PCT = 0.30          # 30% gain triggers take-profit
MAX_OPEN_POSITIONS = 15
MIN_EDGE_SCORE = 70


# ---------------------------------------------------------------------------
# Position Sizing: Half-Kelly
# ---------------------------------------------------------------------------

def kelly_size(
    edge: float,
    price: float,
    bankroll: float,
    max_pct: float = MAX_POSITION_PCT,
) -> float:
    """
    Calculate position size using Half-Kelly criterion.

    Kelly fraction = edge / odds
    We use HALF-Kelly for safety (reduces variance by 75% with only 25% less growth).

    Args:
        edge: estimated edge (probability - price, after fees)
        price: entry price
        bankroll: current bankroll
        max_pct: maximum fraction of bankroll per position

    Returns:
        Position size in USDC
    """
    if price <= 0 or price >= 1 or edge <= 0:
        return 0

    odds = (1 / price) - 1  # decimal odds - 1
    if odds <= 0:
        return 0

    kelly_fraction = edge / odds
    half_kelly = kelly_fraction * 0.5

    # Cap at max_pct of bankroll
    position = min(half_kelly, max_pct) * bankroll

    # Minimum position size
    return max(0, round(position, 2))


# ---------------------------------------------------------------------------
# Paper Trading State
# ---------------------------------------------------------------------------

class PaperTradingState:
    """Persistent state for paper trading."""

    def __init__(self, state_file: Path = PAPER_STATE_FILE):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return {
            "bankroll": DEFAULT_BANKROLL,
            "initial_bankroll": DEFAULT_BANKROLL,
            "positions": {},
            "closed_trades": [],
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "peak_bankroll": DEFAULT_BANKROLL,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def save(self):
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _log_trade(self, trade: dict):
        trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(PAPER_LOG_FILE, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")

    @property
    def bankroll(self) -> float:
        return self.state["bankroll"]

    @bankroll.setter
    def bankroll(self, value: float):
        self.state["bankroll"] = round(value, 2)
        # Track peak and drawdown
        if value > self.state["stats"]["peak_bankroll"]:
            self.state["stats"]["peak_bankroll"] = round(value, 2)
        drawdown = (self.state["stats"]["peak_bankroll"] - value) / self.state["stats"]["peak_bankroll"]
        if drawdown > self.state["stats"]["max_drawdown"]:
            self.state["stats"]["max_drawdown"] = round(drawdown, 4)

    @property
    def positions(self) -> dict:
        return self.state["positions"]

    def open_position(
        self,
        slug: str,
        question: str,
        side: str,
        entry_price: float,
        size_usdc: float,
        edge_score: int,
        signal_type: str,
        token_id: str = "",
    ):
        """Open a new paper position."""
        # Calculate fee (taker for market entry)
        fee = TAKER_FEE * entry_price * (1 - entry_price) * size_usdc
        contracts = size_usdc / entry_price if entry_price > 0 else 0

        position = {
            "slug": slug,
            "question": question[:100],
            "side": side,
            "entry_price": round(entry_price, 4),
            "size_usdc": round(size_usdc, 2),
            "contracts": round(contracts, 2),
            "entry_fee": round(fee, 4),
            "token_id": token_id,
            "edge_score": edge_score,
            "signal_type": signal_type,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "stop_loss": round(entry_price * (1 - STOP_LOSS_PCT), 4) if side == "YES" else round(entry_price * (1 + STOP_LOSS_PCT), 4),
            "take_profit": round(entry_price * (1 + TAKE_PROFIT_PCT), 4) if side == "YES" else round(entry_price * (1 - TAKE_PROFIT_PCT), 4),
        }

        self.state["positions"][slug] = position
        self.bankroll = self.bankroll - size_usdc - fee
        self.state["stats"]["total_trades"] += 1

        self._log_trade({
            "action": "OPEN",
            "slug": slug,
            "side": side,
            "price": entry_price,
            "size_usdc": size_usdc,
            "contracts": contracts,
            "fee": fee,
            "edge_score": edge_score,
            "signal_type": signal_type,
        })

        self.save()
        logger.info(
            f"[PAPER] OPEN: {side} {slug} @ {entry_price:.3f} "
            f"x ${size_usdc:.0f} (score={edge_score}, {signal_type})"
        )

    def close_position(
        self,
        slug: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[dict]:
        """Close a paper position and calculate P&L."""
        pos = self.state["positions"].get(slug)
        if not pos or pos["status"] != "open":
            logger.warning(f"No open position for {slug}")
            return None

        entry_price = pos["entry_price"]
        size_usdc = pos["size_usdc"]
        contracts = pos["contracts"]
        side = pos["side"]

        # Calculate P&L
        if side == "YES":
            raw_pnl = (exit_price - entry_price) * contracts
        else:
            raw_pnl = (entry_price - exit_price) * contracts

        # Exit fee (taker)
        exit_fee = TAKER_FEE * exit_price * (1 - exit_price) * abs(contracts * exit_price)
        net_pnl = raw_pnl - pos["entry_fee"] - exit_fee

        # Update position
        pos["exit_price"] = round(exit_price, 4)
        pos["exit_time"] = datetime.now(timezone.utc).isoformat()
        pos["exit_fee"] = round(exit_fee, 4)
        pos["pnl"] = round(net_pnl, 4)
        pos["exit_reason"] = reason
        pos["status"] = "closed"
        pos["return_pct"] = round(net_pnl / size_usdc * 100, 2) if size_usdc > 0 else 0

        # Update stats
        self.bankroll = self.bankroll + size_usdc + net_pnl
        if net_pnl > 0:
            self.state["stats"]["wins"] += 1
        else:
            self.state["stats"]["losses"] += 1
        self.state["stats"]["total_pnl"] = round(
            self.state["stats"]["total_pnl"] + net_pnl, 4
        )

        # Move to closed trades
        self.state["closed_trades"].append(pos.copy())
        # Keep only last 200 closed trades
        if len(self.state["closed_trades"]) > 200:
            self.state["closed_trades"] = self.state["closed_trades"][-200:]

        del self.state["positions"][slug]

        self._log_trade({
            "action": "CLOSE",
            "slug": slug,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usdc": size_usdc,
            "pnl": net_pnl,
            "return_pct": pos["return_pct"],
            "reason": reason,
        })

        self.save()
        logger.info(
            f"[PAPER] CLOSE: {slug} @ {exit_price:.3f} | "
            f"P&L: ${net_pnl:+.2f} ({pos['return_pct']:+.1f}%) | {reason}"
        )
        return pos

    def get_open_count(self) -> int:
        return sum(1 for p in self.state["positions"].values() if p["status"] == "open")

    def get_total_exposure(self) -> float:
        return sum(
            p["size_usdc"] for p in self.state["positions"].values()
            if p["status"] == "open"
        )

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        stats = self.state["stats"]
        total_trades = stats["total_trades"]
        wins = stats["wins"]
        losses = stats["losses"]
        closed = wins + losses

        return {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": self.state["initial_bankroll"],
            "return_pct": round(
                (self.bankroll - self.state["initial_bankroll"])
                / self.state["initial_bankroll"] * 100, 2
            ),
            "open_positions": self.get_open_count(),
            "total_exposure": round(self.get_total_exposure(), 2),
            "total_trades": total_trades,
            "closed_trades": closed,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(1, closed), 3),
            "total_pnl": round(stats["total_pnl"], 2),
            "max_drawdown": round(stats["max_drawdown"] * 100, 2),
            "peak_bankroll": stats["peak_bankroll"],
        }


# ---------------------------------------------------------------------------
# Paper Trader
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Paper trading engine.
    Uses EdgeDetector for signals and PaperTradingState for tracking.
    """

    def __init__(
        self,
        min_score: int = MIN_EDGE_SCORE,
        use_llm: bool = True,
        max_positions: int = MAX_OPEN_POSITIONS,
    ):
        self.state = PaperTradingState()
        self.detector = EdgeDetector(use_llm=use_llm, min_score=min_score)
        self.scanner = PolymarketScanner()
        self.min_score = min_score
        self.max_positions = max_positions

    def check_exits(self):
        """Check all open positions for stop-loss or take-profit."""
        for slug, pos in list(self.state.positions.items()):
            if pos["status"] != "open":
                continue

            # Get current price
            history = load_price_history(slug, hours=1)
            if history:
                current_price = history[-1]["p"]
            else:
                # Try to get from API
                try:
                    token_id = pos.get("token_id", "")
                    if token_id:
                        mid = self.scanner.get_midpoint(token_id)
                        if mid is not None:
                            current_price = mid
                        else:
                            continue
                    else:
                        continue
                except Exception:
                    continue

            side = pos["side"]
            entry = pos["entry_price"]

            # For NO positions, entry_price is stored as (1 - yes_price),
            # so convert current YES price to NO price for consistent comparison
            if side == "NO":
                current_price = 1 - current_price

            # Check stop-loss
            if side == "YES" and current_price <= pos["stop_loss"]:
                self.state.close_position(slug, current_price, reason="stop_loss")
                continue
            elif side == "NO" and current_price >= pos["stop_loss"]:
                self.state.close_position(slug, current_price, reason="stop_loss")
                continue

            # Check take-profit
            if side == "YES" and current_price >= pos["take_profit"]:
                self.state.close_position(slug, current_price, reason="take_profit")
                continue
            elif side == "NO" and current_price <= pos["take_profit"]:
                self.state.close_position(slug, current_price, reason="take_profit")
                continue

            # Check if position is too old (>7 days) - auto-exit
            entry_time = datetime.fromisoformat(pos["entry_time"])
            if datetime.now(timezone.utc) - entry_time > timedelta(days=7):
                self.state.close_position(slug, current_price, reason="max_holding_period")

    def find_entries(self) -> list:
        """
        Scan for new entry opportunities using edge detector.
        Returns list of opportunities that pass filters.
        """
        # Check if we can open more positions
        if self.state.get_open_count() >= self.max_positions:
            logger.info(f"Max positions ({self.max_positions}) reached, skipping entries")
            return []

        # Get edges from detector
        edges = self.detector.get_actionable_edges(min_score=self.min_score)

        if not edges:
            # Run a fresh scan
            logger.info("No cached edges, running fresh scan...")
            scan = self.detector.scan(limit=200, min_score=self.min_score)
            edges = scan.get("opportunities", [])

        # Filter out markets we already have positions in
        existing_slugs = set(self.state.positions.keys())
        new_edges = [e for e in edges if e["slug"] not in existing_slugs]

        return new_edges

    def execute_entries(self, edges: list) -> list:
        """Execute paper trades for qualified edges."""
        trades = []

        for edge in edges:
            if self.state.get_open_count() >= self.max_positions:
                break

            slug = edge["slug"]
            score = edge["composite_score"]
            direction = edge["direction"]
            price = edge["yes_price"]
            edge_est = edge.get("edge_estimate", 0)

            if direction == "NO_EDGE" or direction == "NEUTRAL":
                continue

            # Determine entry side and price
            if direction == "BUY_YES":
                side = "YES"
                entry_price = price
                token_id = edge.get("yes_token", "")
            elif direction == "BUY_NO":
                side = "NO"
                entry_price = 1 - price
                token_id = edge.get("no_token", "")
            else:
                continue

            # Calculate position size (half-Kelly)
            size = kelly_size(
                edge=edge_est,
                price=entry_price,
                bankroll=self.state.bankroll,
            )

            if size < 5:  # Minimum $5 position
                logger.debug(f"Size too small for {slug} (${size:.2f}), skipping")
                continue

            # Get primary signal type
            signals = edge.get("active_signals", [])
            signal_type = signals[0] if signals else "unknown"

            self.state.open_position(
                slug=slug,
                question=edge.get("question", slug),
                side=side,
                entry_price=entry_price,
                size_usdc=size,
                edge_score=score,
                signal_type=signal_type,
                token_id=token_id,
            )

            trades.append({
                "slug": slug,
                "side": side,
                "price": entry_price,
                "size": size,
                "score": score,
                "signal_type": signal_type,
            })

        return trades

    def run_cycle(self) -> dict:
        """
        Run one full trading cycle:
        1. Check exits (stop-loss, take-profit)
        2. Find new entries
        3. Execute entries
        4. Return summary
        """
        logger.info("--- Paper Trading Cycle ---")

        # 1. Check exits
        self.check_exits()

        # 2. Find entries
        edges = self.find_entries()
        logger.info(f"Found {len(edges)} actionable edges")

        # 3. Execute entries
        trades = self.execute_entries(edges)
        if trades:
            logger.info(f"Opened {len(trades)} new positions")

        # 4. Summary
        summary = self.state.get_summary()
        logger.info(
            f"Portfolio: ${summary['bankroll']:.0f} | "
            f"{summary['open_positions']} open | "
            f"P&L: ${summary['total_pnl']:+.2f} | "
            f"WR: {summary['win_rate']:.0%}"
        )

        return {
            "new_trades": len(trades),
            "trades": trades,
            **summary,
        }

    def run_loop(self, interval_sec: int = 1800):
        """
        Continuous paper trading loop.
        Default interval: 30 minutes.
        """
        logger.info(
            f"Paper trader starting. Bankroll: ${self.state.bankroll:.0f}, "
            f"Min score: {self.min_score}, Max positions: {self.max_positions}"
        )

        cycle = 0
        while True:
            cycle += 1
            try:
                result = self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Paper trader stopped by user")
                break
            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}", exc_info=True)

            logger.info(f"Next cycle in {interval_sec}s...")
            try:
                time.sleep(interval_sec)
            except KeyboardInterrupt:
                logger.info("Paper trader stopped by user")
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_status():
    """Show paper trading status."""
    state = PaperTradingState()
    summary = state.get_summary()

    print(f"\n{'='*60}")
    print(f"  POLYMARKET PAPER TRADER — STATUS")
    print(f"{'='*60}")
    print(f"  Bankroll:      ${summary['bankroll']:,.2f} (started: ${summary['initial_bankroll']:,.0f})")
    print(f"  Return:        {summary['return_pct']:+.2f}%")
    print(f"  Total P&L:     ${summary['total_pnl']:+,.2f}")
    print(f"  Max Drawdown:  {summary['max_drawdown']:.2f}%")
    print(f"  Peak Bankroll: ${summary['peak_bankroll']:,.2f}")
    print(f"\n  Open Positions: {summary['open_positions']} (${summary['total_exposure']:,.0f} exposure)")
    print(f"  Total Trades:   {summary['total_trades']}")
    print(f"  Win Rate:       {summary['win_rate']:.0%} ({summary['wins']}W / {summary['losses']}L)")

    # Show open positions
    positions = state.positions
    if positions:
        print(f"\n  {'OPEN POSITIONS':}")
        print(f"  {'Slug':<35} {'Side':<5} {'Entry':<7} {'Size':<8} {'Score':<6} {'Signal'}")
        print(f"  {'-'*80}")
        for slug, pos in positions.items():
            if pos["status"] == "open":
                print(
                    f"  {slug[:34]:<35} {pos['side']:<5} "
                    f"{pos['entry_price']:.3f}  ${pos['size_usdc']:<7.0f} "
                    f"{pos['edge_score']:<6} {pos['signal_type']}"
                )

    print(f"{'='*60}\n")


def cmd_history(n: int = 20):
    """Show recent trade history."""
    state = PaperTradingState()
    closed = state.state.get("closed_trades", [])

    print(f"\n{'='*70}")
    print(f"  PAPER TRADE HISTORY (last {n})")
    print(f"{'='*70}")

    if not closed:
        print("  No closed trades yet.")
    else:
        for trade in closed[-n:]:
            pnl = trade.get("pnl", 0)
            ret = trade.get("return_pct", 0)
            emoji = "+" if pnl >= 0 else ""
            print(
                f"  {trade.get('exit_time', '?')[:16]} | "
                f"{trade['side']:<4} {trade['slug'][:30]:<31} | "
                f"${pnl:{emoji}.2f} ({ret:+.1f}%) | "
                f"{trade.get('exit_reason', '?')}"
            )

    summary = state.get_summary()
    print(f"\n  Total: {summary['closed_trades']} trades | "
          f"P&L: ${summary['total_pnl']:+,.2f} | "
          f"Win Rate: {summary['win_rate']:.0%}")
    print(f"{'='*70}\n")


def cmd_reset():
    """Reset paper trading state."""
    if PAPER_STATE_FILE.exists():
        PAPER_STATE_FILE.unlink()
        print("Paper trading state reset.")
    else:
        print("No state file to reset.")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Paper Trading Simulator")
    parser.add_argument("--run", action="store_true", help="Run one trading cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuous trading loop")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading state")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum edge score (0-100)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM edge detection")
    parser.add_argument("--interval", type=int, default=1800, help="Loop interval in seconds")
    parser.add_argument("--max-positions", type=int, default=15, help="Max open positions")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    if args.history:
        cmd_history()
        return

    if args.reset:
        cmd_reset()
        return

    if args.loop:
        trader = PaperTrader(
            min_score=args.min_score,
            use_llm=not args.no_llm,
            max_positions=args.max_positions,
        )
        trader.run_loop(interval_sec=args.interval)
        return

    if args.run or True:  # Default
        trader = PaperTrader(
            min_score=args.min_score,
            use_llm=not args.no_llm,
            max_positions=args.max_positions,
        )
        result = trader.run_cycle()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
