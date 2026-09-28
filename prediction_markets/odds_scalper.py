#!/usr/bin/env python3
"""
Polymarket Odds Scalper — Profit from odds movement, not outcome prediction.

Core Strategies:
  1. Catalyst Scalping — Buy before known events (games, debates, data releases),
     sell into the volatility spike regardless of direction.
  2. Mean Reversion — When odds spike >5% on news, buy the other side for reversion.
  3. Momentum — When odds are trending steadily, ride the momentum.
  4. Maker Spread Capture — Post both sides with 0% maker fee, capture the spread.
  5. LLM Mispricing — Use reasoning to identify odds that are logically wrong,
     buy and hold until market corrects (don't need to hold until resolution).

Key Insight: We profit from MOVEMENT, not correctness. A position held for
hours/days and sold at better odds is pure alpha regardless of final outcome.

Polymarket fees: 0% maker, 2% taker. So:
  - Placing limit orders (maker) = FREE
  - Hitting existing orders (taker) = 2% * price * (1-price)
  - Maximum taker fee is at price=0.50: 2% * 0.5 * 0.5 = 0.5%
  - At price=0.10: fee = 2% * 0.1 * 0.9 = 0.18%

Usage:
    python odds_scalper.py --scan                    # Find scalping opportunities
    python odds_scalper.py --monitor SLUG            # Monitor a specific market
    python odds_scalper.py --history SLUG --hours 24 # Show price history
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("odds_scalper")
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

PRICE_HISTORY_DIR = DATA_DIR / "price_history"
PRICE_HISTORY_DIR.mkdir(exist_ok=True)

SCALP_STATE_FILE = DATA_DIR / "scalper_state.json"


# ---------------------------------------------------------------------------
# Price History Tracking
# ---------------------------------------------------------------------------

def record_price(slug: str, yes_price: float, volume: float = 0):
    """Record a price point for a market."""
    history_file = PRICE_HISTORY_DIR / f"{slug[:80]}.jsonl"
    entry = {
        "t": datetime.now(timezone.utc).isoformat(),
        "p": round(yes_price, 4),
        "v": round(volume, 0),
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_price_history(slug: str, hours: float = 24) -> list:
    """Load price history for a market within the last N hours."""
    history_file = PRICE_HISTORY_DIR / f"{slug[:80]}.jsonl"
    if not history_file.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    history = []
    with open(history_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = datetime.fromisoformat(entry["t"])
                if ts >= cutoff:
                    history.append(entry)
            except Exception:
                continue
    return history


# ---------------------------------------------------------------------------
# Scalping Signal Detection
# ---------------------------------------------------------------------------

def detect_mean_reversion(history: list, current_price: float) -> dict:
    """
    Detect mean reversion opportunities.
    If price spiked >5% from recent average, it often reverts.
    """
    if len(history) < 10:
        return {"signal": "INSUFFICIENT_DATA"}

    prices = [h["p"] for h in history]
    avg = sum(prices) / len(prices)
    std = (sum((p - avg) ** 2 for p in prices) / len(prices)) ** 0.5

    if std < 0.005:
        return {"signal": "LOW_VOL", "avg": avg, "std": std}

    z_score = (current_price - avg) / std if std > 0 else 0

    if z_score > 2.0:
        # Price spiked UP — mean reversion says BUY NO
        return {
            "signal": "MEAN_REVERT_SHORT",
            "direction": "BUY_NO",
            "z_score": round(z_score, 2),
            "avg": round(avg, 4),
            "current": round(current_price, 4),
            "expected_revert": round(avg + std, 4),
            "edge_estimate": round(current_price - avg - std, 4),
        }
    elif z_score < -2.0:
        # Price spiked DOWN — mean reversion says BUY YES
        return {
            "signal": "MEAN_REVERT_LONG",
            "direction": "BUY_YES",
            "z_score": round(z_score, 2),
            "avg": round(avg, 4),
            "current": round(current_price, 4),
            "expected_revert": round(avg - std, 4),
            "edge_estimate": round(avg - std - current_price, 4),
        }

    return {"signal": "NO_SIGNAL", "z_score": round(z_score, 2), "avg": round(avg, 4)}


def detect_momentum(history: list, current_price: float) -> dict:
    """
    Detect momentum signals.
    If price has been trending steadily in one direction, ride it.
    """
    if len(history) < 20:
        return {"signal": "INSUFFICIENT_DATA"}

    prices = [h["p"] for h in history]

    # Check last N price changes
    recent = prices[-20:]
    changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    up_count = sum(1 for c in changes if c > 0.001)
    down_count = sum(1 for c in changes if c < -0.001)
    total_move = recent[-1] - recent[0]

    # Strong momentum: >70% of moves in same direction AND >3% total move
    if up_count > len(changes) * 0.7 and total_move > 0.03:
        return {
            "signal": "MOMENTUM_UP",
            "direction": "BUY_YES",
            "move": round(total_move, 4),
            "consistency": round(up_count / len(changes), 2),
            "edge_estimate": round(total_move * 0.3, 4),  # Expect 30% continuation
        }
    elif down_count > len(changes) * 0.7 and total_move < -0.03:
        return {
            "signal": "MOMENTUM_DOWN",
            "direction": "BUY_NO",
            "move": round(total_move, 4),
            "consistency": round(down_count / len(changes), 2),
            "edge_estimate": round(abs(total_move) * 0.3, 4),
        }

    return {"signal": "NO_MOMENTUM", "total_move": round(total_move, 4)}


def detect_spread_opportunity(orderbook: dict) -> dict:
    """
    Detect wide spreads where maker quoting is profitable.
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        return {"signal": "NO_BOOK"}

    best_bid = float(bids[0].get("price", 0))
    best_ask = float(asks[0].get("price", 1))
    spread = best_ask - best_bid

    # Maker quoting is profitable when spread > 2 * minimum tick
    # On Polymarket, minimum tick is $0.01
    if spread >= 0.03:  # 3+ cent spread = profitable market making
        mid = (best_bid + best_ask) / 2
        our_bid = round(best_bid + 0.01, 2)
        our_ask = round(best_ask - 0.01, 2)
        capture = our_ask - our_bid

        return {
            "signal": "WIDE_SPREAD",
            "spread": round(spread, 4),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "our_bid": our_bid,
            "our_ask": our_ask,
            "capture_per_contract": round(capture, 4),
            "bid_depth": sum(float(b.get("size", 0)) for b in bids[:3]),
            "ask_depth": sum(float(a.get("size", 0)) for a in asks[:3]),
        }

    return {"signal": "TIGHT_SPREAD", "spread": round(spread, 4)}


def calculate_taker_fee(price: float) -> float:
    """Calculate Polymarket taker fee for a given price."""
    return TAKER_FEE * price * (1 - price)


def calculate_scalp_breakeven(entry_price: float, direction: str = "YES") -> float:
    """
    Calculate the minimum price move needed to break even on a scalp.
    Assumes taker fee on both entry and exit.
    """
    entry_fee = calculate_taker_fee(entry_price)

    if direction == "YES":
        # Need price to go UP
        # At exit price p_exit: fee = TAKER_FEE * p_exit * (1 - p_exit)
        # Profit = (p_exit - entry_price) - entry_fee - exit_fee
        # For breakeven: p_exit - entry_price = entry_fee + exit_fee
        # Approximate: need move > 2 * max_fee ≈ 2 * 0.5% = 1%
        # But fee decreases away from 0.5, so typically need ~0.5-1% move
        exit_price = entry_price + entry_fee * 2  # rough estimate
        return round(exit_price - entry_price, 4)
    else:
        exit_price = entry_price - entry_fee * 2
        return round(entry_price - exit_price, 4)


# ---------------------------------------------------------------------------
# Scalping Scanner
# ---------------------------------------------------------------------------

class OddsScalper:
    """
    Scans Polymarket for scalping opportunities.
    """

    def __init__(self):
        self.scanner = PolymarketScanner()

    def scan_all(self, limit: int = 500) -> dict:
        """
        Scan all active markets for scalping opportunities.
        No LLM needed — pure quantitative signals.
        """
        logger.info(f"Scanning {limit} markets for scalping opportunities...")

        markets = self.scanner.get_active_markets(limit=limit)
        logger.info(f"Fetched {len(markets)} markets")

        opportunities = []
        spreads = []
        market_summaries = []

        for m in markets:
            prices = self.scanner._parse_prices(m.get("outcomePrices", []))
            if not prices:
                continue

            yes_price = prices[0]
            vol = float(m.get("volumeNum", 0) or 0)
            slug = m.get("slug", "")
            question = m.get("question", "")

            if not (0.05 <= yes_price <= 0.95) or vol < 10_000:
                continue

            # Record current price
            record_price(slug, yes_price, vol)

            # Load history and check signals
            history = load_price_history(slug, hours=48)

            summary = {
                "question": question[:100],
                "slug": slug,
                "price": round(yes_price, 3),
                "volume": vol,
                "history_points": len(history),
                "breakeven_move": calculate_scalp_breakeven(yes_price),
                "taker_fee": round(calculate_taker_fee(yes_price), 4),
            }

            # Mean reversion signal
            mr = detect_mean_reversion(history, yes_price)
            if mr.get("signal") in ("MEAN_REVERT_LONG", "MEAN_REVERT_SHORT"):
                summary["mean_reversion"] = mr
                opportunities.append({
                    "type": "mean_reversion",
                    "question": question[:100],
                    "slug": slug,
                    "price": yes_price,
                    "volume": vol,
                    **mr,
                })

            # Momentum signal
            mom = detect_momentum(history, yes_price)
            if mom.get("signal") in ("MOMENTUM_UP", "MOMENTUM_DOWN"):
                summary["momentum"] = mom
                opportunities.append({
                    "type": "momentum",
                    "question": question[:100],
                    "slug": slug,
                    "price": yes_price,
                    "volume": vol,
                    **mom,
                })

            # Spread opportunity (check orderbook for high-volume markets only to avoid rate limits)
            if vol > 500_000:
                tokens = m.get("clobTokenIds", [])
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except Exception:
                        tokens = []

                if tokens:
                    book = self.scanner.get_orderbook(tokens[0])
                    spread_sig = detect_spread_opportunity(book)
                    if spread_sig.get("signal") == "WIDE_SPREAD":
                        summary["spread"] = spread_sig
                        spreads.append({
                            "question": question[:100],
                            "slug": slug,
                            "price": yes_price,
                            "volume": vol,
                            "token_id": tokens[0],
                            **spread_sig,
                        })
                    time.sleep(0.3)  # Rate limit orderbook calls

            market_summaries.append(summary)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_markets": len(markets),
            "scanned": len(market_summaries),
            "mean_reversion_signals": len([o for o in opportunities if o["type"] == "mean_reversion"]),
            "momentum_signals": len([o for o in opportunities if o["type"] == "momentum"]),
            "wide_spread_markets": len(spreads),
            "opportunities": opportunities,
            "spread_opportunities": spreads,
            "market_summaries": market_summaries[:20],  # Top 20 by volume
        }

        # Save state
        with open(SCALP_STATE_FILE, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def format_report(self, result: dict) -> str:
        """Format scan results for display."""
        lines = [
            f"{'='*60}",
            f"  POLYMARKET SCALPING SCAN",
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"{'='*60}",
            f"  Markets scanned: {result['scanned']}",
            f"  Mean reversion signals: {result['mean_reversion_signals']}",
            f"  Momentum signals: {result['momentum_signals']}",
            f"  Wide spread markets: {result['wide_spread_markets']}",
        ]

        opps = result.get("opportunities", [])
        if opps:
            lines.append(f"\n  TRADING SIGNALS:")
            for o in opps:
                lines.append(
                    f"    [{o['type'].upper()}] {o['direction']} "
                    f"{o['question'][:50]}"
                )
                lines.append(
                    f"      Price: {o['price']:.3f} | Vol: ${o['volume']:,.0f} | "
                    f"Edge est: {o.get('edge_estimate', 0):.3f}"
                )

        spreads = result.get("spread_opportunities", [])
        if spreads:
            lines.append(f"\n  SPREAD CAPTURE:")
            for s in spreads:
                lines.append(
                    f"    {s['question'][:50]}")
                lines.append(
                    f"      Spread: {s['spread']:.3f} | "
                    f"Our capture: {s['capture_per_contract']:.3f}/contract"
                )

        if not opps and not spreads:
            lines.append("\n  No scalping signals detected. Need more price history.")
            lines.append("  Run this scan periodically to build history.")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket Odds Scalper")
    parser.add_argument("--scan", action="store_true", help="Scan for scalping opportunities")
    parser.add_argument("--monitor", type=str, help="Monitor a specific market slug")
    parser.add_argument("--history", type=str, help="Show price history for a slug")
    parser.add_argument("--hours", type=float, default=24, help="Hours of history to show")
    parser.add_argument("--limit", type=int, default=500, help="Max markets to scan")
    args = parser.parse_args()

    if args.history:
        history = load_price_history(args.history, hours=args.hours)
        if not history:
            print(f"No history for '{args.history}'. Run --scan first to collect data.")
            return
        print(f"\nPrice history for '{args.history}' (last {args.hours}h):")
        for h in history[-20:]:
            print(f"  {h['t'][:19]} | {h['p']:.3f}")
        return

    if args.scan or True:  # Default to scan
        scalper = OddsScalper()
        result = scalper.scan_all(limit=args.limit)
        report = scalper.format_report(result)
        print(report.encode('ascii', errors='replace').decode('ascii'))


if __name__ == "__main__":
    main()
