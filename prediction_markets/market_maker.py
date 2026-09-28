#!/usr/bin/env python3
"""
Polymarket Market Maker — Automated two-sided quoting with inventory management.

Strategies implemented:
  1. LLM Fair Value MM — Quote around LLM-estimated fair value
  2. Vol-Informed MM — Use our vol model (IC=0.644) to price event brackets
  3. Combinatorial Arb — Exploit logical mispricings across related markets
  4. Directional Edge — Take positions when scanner finds strong edge

Key features:
  - 0% maker fees on Polymarket (our edge)
  - Dynamic spread based on inventory, volatility, and market depth
  - Half-Kelly position sizing
  - Risk limits: per-market, total exposure, max loss
  - Discord reporting at each cycle

Usage:
    python market_maker.py --mode paper          # Paper trading (log only)
    python market_maker.py --mode live           # Live trading (requires wallet)
    python market_maker.py --mode paper --once   # Single cycle then exit
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("market_maker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from polymarket_client import PolymarketScanner, PolymarketTrader, TAKER_FEE, MAKER_FEE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MMConfig:
    """Market maker configuration."""
    # Risk limits
    max_position_per_market: float = 200.0    # Max USDC per market
    max_total_exposure: float = 2000.0        # Max total USDC deployed
    max_daily_loss: float = 100.0             # Stop trading if daily loss exceeds this
    max_open_markets: int = 10                # Max simultaneous markets

    # Quoting parameters
    base_spread: float = 0.03                 # Base spread (3 cents each side)
    min_spread: float = 0.02                  # Minimum spread (2 cents)
    max_spread: float = 0.08                  # Maximum spread (8 cents)
    inventory_skew_factor: float = 0.5        # How much inventory skews quotes
    quote_size_usdc: float = 25.0             # Default quote size per side

    # Edge thresholds
    min_edge_to_trade: float = 0.015          # Min edge to take directional position
    min_edge_mm: float = 0.005                # Min spread capture for MM

    # Timing
    cycle_interval_seconds: int = 300         # 5 minutes between cycles
    order_ttl_seconds: int = 600              # Cancel orders after 10 min

    # Position sizing
    kelly_fraction: float = 0.15              # Fraction of Kelly criterion

    # Mode
    mode: str = "paper"                       # "paper" or "live"


# ---------------------------------------------------------------------------
# Position & Order Tracking
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Tracks a position in a single market."""
    market_slug: str
    question: str
    token_id: str
    side: str                    # "YES" or "NO"
    avg_entry: float             # Average entry price
    size_usdc: float             # Total USDC deployed
    contracts: float             # Number of contracts
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: str = ""
    last_price: float = 0.0

    @property
    def total_pnl(self):
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class Order:
    """Tracks a pending order."""
    order_id: str
    market_slug: str
    token_id: str
    side: str                    # "BUY" or "SELL"
    outcome: str                 # "YES" or "NO"
    price: float
    size: float                  # In contracts
    created_at: str
    status: str = "pending"      # "pending", "filled", "cancelled"


@dataclass
class MMState:
    """Full market maker state."""
    positions: dict = field(default_factory=dict)   # market_slug -> Position
    open_orders: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    total_exposure: float = 0.0
    cycle_count: int = 0
    started_at: str = ""
    last_cycle: str = ""
    is_stopped: bool = False
    stop_reason: str = ""

    def to_dict(self):
        d = asdict(self)
        # Convert Position objects
        d["positions"] = {k: asdict(v) if isinstance(v, Position) else v
                          for k, v in self.positions.items()}
        return d


STATE_FILE = DATA_DIR / "mm_state.json"


def load_state() -> MMState:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            state = MMState()
            state.positions = {
                k: Position(**v) if isinstance(v, dict) else v
                for k, v in data.get("positions", {}).items()
            }
            state.daily_pnl = data.get("daily_pnl", 0.0)
            state.total_pnl = data.get("total_pnl", 0.0)
            state.total_exposure = data.get("total_exposure", 0.0)
            state.cycle_count = data.get("cycle_count", 0)
            state.started_at = data.get("started_at", "")
            state.last_cycle = data.get("last_cycle", "")
            state.trade_log = data.get("trade_log", [])[-500:]  # Keep last 500
            state.is_stopped = data.get("is_stopped", False)
            state.stop_reason = data.get("stop_reason", "")
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return MMState(started_at=datetime.now(timezone.utc).isoformat())


def save_state(state: MMState):
    with open(STATE_FILE, "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Market Maker Engine
# ---------------------------------------------------------------------------

class MarketMaker:
    """
    Polymarket Market Maker.

    Each cycle:
    1. Scan for opportunities (financial markets with edge)
    2. Update existing positions (mark-to-market)
    3. Cancel stale orders
    4. Place new two-sided quotes on selected markets
    5. Take directional positions on high-edge opportunities
    6. Enforce risk limits
    7. Report to Discord
    """

    def __init__(self, config: MMConfig = None):
        self.config = config or MMConfig()
        self.scanner = PolymarketScanner()
        self.trader = PolymarketTrader(mode=self.config.mode)
        self.state = load_state()
        self._last_engine_scan = 0  # epoch time of last strategy engine scan
        self._cached_engine_opps = []  # cached engine opportunities

        # Fair value estimator (lazy load)
        self._llm_estimator = None

    def _get_fair_value_fn(self):
        """Get LLM fair value function if available."""
        if self._llm_estimator is None:
            try:
                from llm_fair_value import LLMFairValueEstimator
                self._llm_estimator = LLMFairValueEstimator()
                return self._llm_estimator.as_fair_value_fn("fast")
            except Exception:
                return None
        return self._llm_estimator.as_fair_value_fn("fast") if self._llm_estimator else None

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Run one complete MM cycle. Returns cycle report dict."""
        cycle_start = time.time()

        # Reset daily PnL at start of new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        started_date = (self.state.started_at or "")[:10]
        last_cycle_date = (self.state.last_cycle or "")[:10]
        if last_cycle_date and last_cycle_date != today:
            logger.info(f"New day detected ({last_cycle_date} -> {today}), resetting daily PnL")
            self.state.daily_pnl = 0.0
            self.state.is_stopped = False
            self.state.stop_reason = ""

        self.state.cycle_count += 1
        self.state.last_cycle = datetime.now(timezone.utc).isoformat()

        report = {
            "cycle": self.state.cycle_count,
            "timestamp": self.state.last_cycle,
            "mode": self.config.mode,
            "actions": [],
            "errors": [],
            "positions": {},
            "daily_pnl": 0.0,
            "total_exposure": 0.0,
        }

        # Check if stopped
        if self.state.is_stopped:
            report["actions"].append(f"STOPPED: {self.state.stop_reason}")
            return report

        # Check daily loss limit
        if self.state.daily_pnl < -self.config.max_daily_loss:
            self.state.is_stopped = True
            self.state.stop_reason = f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"
            report["actions"].append(f"STOPPED: {self.state.stop_reason}")
            save_state(self.state)
            return report

        try:
            # Step 1: Scan for opportunities
            opportunities = self._scan_opportunities()
            report["actions"].append(f"Scanned: {len(opportunities)} opportunities found")

            # Step 2: Update existing positions
            self._update_positions()

            # Step 3: Cancel stale orders
            cancelled = self._cancel_stale_orders()
            if cancelled:
                report["actions"].append(f"Cancelled {cancelled} stale orders")

            # Step 4: Evaluate and act on opportunities
            new_trades = self._evaluate_and_trade(opportunities)
            for trade in new_trades:
                report["actions"].append(
                    f"{trade['action']} {trade['side']} on '{trade['question'][:40]}' "
                    f"@ {trade['price']:.3f} (edge: {trade['edge']:.3f})"
                )

            # Step 5: Risk check
            self._enforce_risk_limits()

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            report["errors"].append(str(e))

        # Update report
        report["positions"] = {
            slug: {
                "side": pos.side,
                "size": pos.size_usdc,
                "entry": pos.avg_entry,
                "last": pos.last_price,
                "pnl": pos.total_pnl,
            }
            for slug, pos in self.state.positions.items()
        }
        report["daily_pnl"] = round(self.state.daily_pnl, 2)
        report["total_pnl"] = round(self.state.total_pnl, 2)
        report["total_exposure"] = round(self.state.total_exposure, 2)
        report["duration_seconds"] = round(time.time() - cycle_start, 1)

        save_state(self.state)
        return report

    # ------------------------------------------------------------------
    # Step 1: Scan
    # ------------------------------------------------------------------

    def _scan_opportunities(self) -> list:
        """Scan Polymarket for tradeable opportunities."""
        try:
            fv_fn = self._get_fair_value_fn()
            opps = self.scanner.scan(
                limit=200,
                min_net_edge=self.config.min_edge_mm,
                fair_value_fn=fv_fn,
            )

            # Also check combinatorial mispricings across ALL markets (not just financial)
            markets = self.scanner.get_active_markets(limit=500)
            mispricings = self.scanner.find_mispricings(markets)

            for mp in mispricings:
                if mp["edge_per_contract"] > self.config.min_edge_to_trade:
                    opps.append({
                        "question": f"Arb: {mp['group']}",
                        "market_slug": mp["group"],
                        "net_edge": mp["edge_per_contract"],
                        "action": "SELL" if mp["direction"] == "overpriced" else "BUY",
                        "fair_value": 1.0 - mp["probability_sum"] if mp["direction"] == "overpriced" else mp["probability_sum"],
                        "market_price": mp["probability_sum"],
                        "side": "NO" if mp["direction"] == "overpriced" else "YES",
                        "source": "combinatorial_arb",
                        "token_id": None,
                    })

            # Also load strategy engine opportunities if available (throttled to 1x/hour)
            try:
                now = time.time()
                if now - self._last_engine_scan > 3600:  # Once per hour max
                    from strategy_engine import StrategyEngine
                    engine = StrategyEngine(mode="demo", use_llm=False)
                    eng_opps = engine.scan_all()
                    self._cached_engine_opps = eng_opps.get("top_opportunities", [])[:15]
                    self._last_engine_scan = now
                for eo in self._cached_engine_opps:
                    if eo.get("net_edge_after_fees", 0) > self.config.min_edge_to_trade:
                        opps.append({
                            "question": eo.get("group", eo.get("title", "?")),
                            "market_slug": eo.get("group", ""),
                            "net_edge": eo["net_edge_after_fees"],
                            "action": eo.get("action", "BUY"),
                            "fair_value": None,
                            "market_price": eo.get("market_price", 0),
                            "side": eo.get("side", "YES"),
                            "source": eo.get("strategy", "engine"),
                            "token_id": None,
                        })
            except Exception as e:
                logger.debug(f"Strategy engine unavailable: {e}")

            return opps

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Step 2: Update positions
    # ------------------------------------------------------------------

    def _update_positions(self):
        """Mark-to-market all open positions."""
        total_exposure = 0.0
        for slug, pos in list(self.state.positions.items()):
            try:
                if pos.token_id:
                    mid = self.scanner.get_midpoint(pos.token_id)
                    if mid is not None:
                        pos.last_price = mid
                        if pos.side == "YES":
                            pos.unrealized_pnl = (mid - pos.avg_entry) * pos.contracts
                        else:
                            pos.unrealized_pnl = ((1 - mid) - pos.avg_entry) * pos.contracts
                total_exposure += pos.size_usdc
            except Exception as e:
                logger.warning(f"Position update failed for {slug}: {e}")

        self.state.total_exposure = total_exposure

    # ------------------------------------------------------------------
    # Step 3: Cancel stale orders
    # ------------------------------------------------------------------

    def _cancel_stale_orders(self) -> int:
        """Cancel orders older than TTL. Returns count cancelled."""
        if not self.state.open_orders:
            return 0

        now = datetime.now(timezone.utc)
        cancelled = 0
        remaining = []

        for order in self.state.open_orders:
            try:
                created = datetime.fromisoformat(order.get("created_at", ""))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (now - created).total_seconds()

                if age > self.config.order_ttl_seconds:
                    if self.config.mode == "live" and self.trader.client:
                        try:
                            self.trader.client.cancel(order["order_id"])
                        except Exception as e:
                            logger.warning(f"Cancel failed: {e}")
                    cancelled += 1
                else:
                    remaining.append(order)
            except Exception:
                remaining.append(order)

        self.state.open_orders = remaining
        return cancelled

    # ------------------------------------------------------------------
    # Step 4: Evaluate and trade
    # ------------------------------------------------------------------

    def _evaluate_and_trade(self, opportunities: list) -> list:
        """Evaluate opportunities and execute trades. Returns list of trade dicts."""
        trades = []

        for opp in opportunities:
            slug = opp.get("market_slug", "")
            net_edge = opp.get("net_edge", 0)
            question = opp.get("question", "")
            token_id = opp.get("token_id")

            # Skip if already at max markets
            if len(self.state.positions) >= self.config.max_open_markets:
                if slug not in self.state.positions:
                    continue

            # Skip if already max exposure
            if self.state.total_exposure >= self.config.max_total_exposure:
                if slug not in self.state.positions:
                    continue

            # Skip if already positioned in this market
            if slug in self.state.positions:
                continue

            # Determine action
            if net_edge >= self.config.min_edge_to_trade:
                trade = self._execute_directional(opp)
                if trade:
                    trades.append(trade)
            elif net_edge >= self.config.min_edge_mm:
                trade = self._place_mm_quotes(opp)
                if trade:
                    trades.append(trade)

            # Rate limit
            time.sleep(0.2)

        return trades

    def _compute_size(self, edge: float, price: float) -> float:
        """Compute position size using fractional Kelly."""
        if edge <= 0 or price <= 0.01 or price >= 0.99:
            return 0

        # Kelly fraction for binary outcome
        kelly_full = edge / (1 - price)
        kelly_adj = kelly_full * self.config.kelly_fraction

        # Size in USDC
        size = kelly_adj * self.config.max_position_per_market
        size = max(5.0, min(size, self.config.max_position_per_market))

        # Check remaining exposure budget
        remaining = self.config.max_total_exposure - self.state.total_exposure
        size = min(size, remaining)

        return round(size, 2)

    def _execute_directional(self, opp: dict) -> Optional[dict]:
        """Take a directional position on a high-edge opportunity."""
        side = opp.get("side", "YES")
        price = opp.get("entry_price", opp.get("market_price", 0.5))
        edge = opp.get("net_edge", 0)
        slug = opp.get("market_slug", "")
        question = opp.get("question", "")
        token_id = opp.get("token_id")
        fair_value = opp.get("fair_value", price)

        size_usdc = self._compute_size(edge, price)
        if size_usdc < 5.0:
            return None

        contracts = size_usdc / price if price > 0 else 0

        trade_record = {
            "action": "DIRECTIONAL",
            "side": side,
            "price": price,
            "fair_value": fair_value,
            "edge": edge,
            "size_usdc": size_usdc,
            "contracts": contracts,
            "question": question,
            "market_slug": slug,
            "token_id": token_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
        }

        if self.config.mode == "live" and self.trader.client and token_id:
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY

                order = self.trader.client.create_and_post_order(
                    OrderArgs(
                        token_id=token_id,
                        price=price,
                        size=contracts,
                        side=BUY,
                    )
                )
                trade_record["order_id"] = order.get("orderID", "unknown")
                trade_record["status"] = "placed"
                logger.info(f"LIVE ORDER: {side} {contracts:.1f} contracts @ {price:.3f}")
            except Exception as e:
                trade_record["status"] = "failed"
                trade_record["error"] = str(e)
                logger.error(f"Order placement failed: {e}")
                return trade_record
        else:
            trade_record["status"] = "paper"
            logger.info(
                f"PAPER: {side} ${size_usdc:.0f} on '{question[:40]}' "
                f"@ {price:.3f} (edge: {edge:.3f})"
            )

        # Update state
        self.state.positions[slug] = Position(
            market_slug=slug,
            question=question,
            token_id=token_id or "",
            side=side,
            avg_entry=price,
            size_usdc=size_usdc,
            contracts=contracts,
            opened_at=datetime.now(timezone.utc).isoformat(),
            last_price=price,
        )
        self.state.total_exposure += size_usdc
        self.state.trade_log.append(trade_record)

        return trade_record

    def _place_mm_quotes(self, opp: dict) -> Optional[dict]:
        """Place two-sided market making quotes."""
        fair_value = opp.get("fair_value", 0.5)
        token_id = opp.get("token_id")
        slug = opp.get("market_slug", "")
        question = opp.get("question", "")

        if not token_id:
            return None

        # Dynamic spread based on inventory
        spread = self._compute_spread(slug, fair_value)
        half_spread = spread / 2

        bid_price = max(0.01, round(fair_value - half_spread, 2))
        ask_price = min(0.99, round(fair_value + half_spread, 2))

        # Inventory skew: if we own YES, lower bid (less eager to buy more)
        existing = self.state.positions.get(slug)
        if existing:
            skew = self.config.inventory_skew_factor * (existing.contracts / 100)
            if existing.side == "YES":
                bid_price = max(0.01, bid_price - skew)
                ask_price = min(0.99, ask_price - skew * 0.5)
            else:
                bid_price = max(0.01, bid_price + skew * 0.5)
                ask_price = min(0.99, ask_price + skew)

        quote_size = self.config.quote_size_usdc
        bid_contracts = quote_size / bid_price if bid_price > 0 else 0
        ask_contracts = quote_size / ask_price if ask_price > 0 else 0

        trade_record = {
            "action": "MM_QUOTE",
            "side": "TWO_SIDED",
            "price": fair_value,
            "bid": bid_price,
            "ask": ask_price,
            "spread": spread,
            "edge": spread - 0.001,  # Spread capture minus minimal costs
            "size_usdc": quote_size,
            "question": question,
            "market_slug": slug,
            "token_id": token_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
        }

        if self.config.mode == "live" and self.trader.client:
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY, SELL

                # Place bid (buy YES)
                bid_order = self.trader.client.create_and_post_order(
                    OrderArgs(
                        token_id=token_id,
                        price=bid_price,
                        size=bid_contracts,
                        side=BUY,
                    )
                )
                # Place ask (sell YES / buy NO)
                ask_order = self.trader.client.create_and_post_order(
                    OrderArgs(
                        token_id=token_id,
                        price=ask_price,
                        size=ask_contracts,
                        side=SELL,
                    )
                )

                self.state.open_orders.extend([
                    {"order_id": bid_order.get("orderID", ""), "side": "BUY",
                     "price": bid_price, "market_slug": slug,
                     "created_at": datetime.now(timezone.utc).isoformat()},
                    {"order_id": ask_order.get("orderID", ""), "side": "SELL",
                     "price": ask_price, "market_slug": slug,
                     "created_at": datetime.now(timezone.utc).isoformat()},
                ])
                trade_record["status"] = "placed"
            except Exception as e:
                trade_record["status"] = "failed"
                trade_record["error"] = str(e)
                logger.error(f"MM quote placement failed: {e}")
        else:
            trade_record["status"] = "paper"
            logger.info(
                f"PAPER MM: '{question[:40]}' bid={bid_price:.2f} ask={ask_price:.2f} "
                f"spread={spread:.3f}"
            )

        self.state.trade_log.append(trade_record)
        return trade_record

    def _compute_spread(self, slug: str, fair_value: float) -> float:
        """Compute dynamic spread based on conditions."""
        spread = self.config.base_spread

        # Wider spread near extremes (less confident)
        if fair_value < 0.15 or fair_value > 0.85:
            spread *= 1.5
        elif fair_value < 0.25 or fair_value > 0.75:
            spread *= 1.2

        # Wider spread if we have inventory
        existing = self.state.positions.get(slug)
        if existing and existing.size_usdc > 100:
            spread *= 1.3

        return max(self.config.min_spread, min(self.config.max_spread, spread))

    # ------------------------------------------------------------------
    # Step 5: Risk limits
    # ------------------------------------------------------------------

    def _enforce_risk_limits(self):
        """Check and enforce all risk limits."""
        # Close positions that have hit stop loss
        for slug, pos in list(self.state.positions.items()):
            if pos.unrealized_pnl < -(pos.size_usdc * 0.3):  # 30% stop loss
                logger.warning(f"Stop loss hit on {slug}: PnL={pos.unrealized_pnl:.2f}")
                self._close_position(slug, "stop_loss")

        # Check total exposure
        if self.state.total_exposure > self.config.max_total_exposure * 1.1:
            logger.warning(f"Exposure limit exceeded: ${self.state.total_exposure:.0f}")
            # Close smallest position to reduce exposure
            if self.state.positions:
                smallest = min(self.state.positions.items(),
                             key=lambda x: x[1].size_usdc)
                self._close_position(smallest[0], "exposure_limit")

    def _close_position(self, slug: str, reason: str):
        """Close a position (paper or live)."""
        pos = self.state.positions.get(slug)
        if not pos:
            return

        # In live mode, place a closing order on-exchange
        if self.config.mode == "live" and pos.token_id:
            try:
                close_side = "SELL" if pos.side == "YES" else "BUY"
                self.trader.place_order(
                    token_id=pos.token_id,
                    side=close_side,
                    price=pos.last_price,
                    size=pos.contracts,
                )
                logger.info(f"Placed closing order for {slug}: {close_side} {pos.contracts} @ {pos.last_price:.3f}")
            except Exception as e:
                logger.error(f"Failed to close position on-exchange for {slug}: {e}")
                return  # Don't remove from state if exchange close failed

        pnl = pos.total_pnl
        self.state.daily_pnl += pnl
        self.state.total_pnl += pnl
        self.state.total_exposure -= pos.size_usdc

        self.state.trade_log.append({
            "action": "CLOSE",
            "reason": reason,
            "market_slug": slug,
            "side": pos.side,
            "pnl": round(pnl, 2),
            "size_usdc": pos.size_usdc,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
        })

        del self.state.positions[slug]
        logger.info(f"Closed {slug}: PnL=${pnl:.2f} (reason: {reason})")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_cycle_report(self, report: dict) -> str:
        """Format cycle report for Discord."""
        lines = [
            f"**MM Cycle #{report['cycle']}** ({report['mode'].upper()})",
        ]

        for action in report["actions"][:8]:
            lines.append(f"  {action}")

        if report["positions"]:
            lines.append(f"\n**Positions ({len(report['positions'])}):**")
            for slug, pos in list(report["positions"].items())[:5]:
                lines.append(
                    f"  {slug[:30]}: {pos['side']} ${pos['size']:.0f} "
                    f"@ {pos['entry']:.3f} -> {pos['last']:.3f} "
                    f"(PnL: ${pos['pnl']:.2f})"
                )

        lines.append(
            f"\nDaily P&L: ${report['daily_pnl']:.2f} | "
            f"Total: ${report.get('total_pnl', 0):.2f} | "
            f"Exposure: ${report['total_exposure']:.0f} | "
            f"Duration: {report.get('duration_seconds', 0):.1f}s"
        )

        if report["errors"]:
            lines.append(f"\nErrors: {'; '.join(report['errors'][:3])}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(self, max_cycles: int = 0):
        """Run the market maker in a loop.

        Args:
            max_cycles: Stop after this many cycles (0 = infinite)
        """
        logger.info(f"Market Maker starting in {self.config.mode} mode")
        logger.info(f"Max exposure: ${self.config.max_total_exposure}, "
                    f"Max per market: ${self.config.max_position_per_market}")

        cycle = 0
        while True:
            try:
                report = self.run_cycle()
                discord_msg = self.format_cycle_report(report)

                # Post to Discord via sidecar file
                self._post_discord(discord_msg)

                cycle += 1
                if max_cycles > 0 and cycle >= max_cycles:
                    logger.info(f"Completed {max_cycles} cycles, stopping.")
                    break

                if self.state.is_stopped:
                    logger.warning(f"MM stopped: {self.state.stop_reason}")
                    break

                logger.info(f"Sleeping {self.config.cycle_interval_seconds}s...")
                time.sleep(self.config.cycle_interval_seconds)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(30)

        save_state(self.state)
        logger.info("Market Maker stopped.")

    def _post_discord(self, message: str):
        """Post message to Discord via sidecar file."""
        sidecar = THIS_DIR.parents[1] / ".discord_pending.json"
        try:
            pending = []
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        pending = json.load(f)
                except Exception:
                    pending = []
            pending.append({
                "channel": "prediction-markets",
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            with open(sidecar, "w") as f:
                json.dump(pending, f, indent=2)
        except Exception as e:
            logger.warning(f"Discord sidecar write failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Market Maker")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                       help="Trading mode")
    parser.add_argument("--once", action="store_true",
                       help="Run single cycle then exit")
    parser.add_argument("--max-exposure", type=float, default=2000,
                       help="Max total USDC exposure")
    parser.add_argument("--cycle-interval", type=int, default=300,
                       help="Seconds between cycles")
    parser.add_argument("--reset", action="store_true",
                       help="Reset state file before starting")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("State reset.")

    config = MMConfig(
        mode=args.mode,
        max_total_exposure=args.max_exposure,
        cycle_interval_seconds=args.cycle_interval,
    )

    mm = MarketMaker(config)

    if args.once:
        report = mm.run_cycle()
        print(mm.format_cycle_report(report))
    else:
        mm.run_loop()


if __name__ == "__main__":
    main()
