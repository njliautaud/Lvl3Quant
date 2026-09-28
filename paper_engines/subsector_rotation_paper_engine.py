#!/usr/bin/env python3
"""
Sub-Sector Rotation Paper Trading Engine
==========================================
Trades rotation signals from the sub-sector tracker and ML model.

Strategy:
  - Go long ETF proxies for top-ranked sub-sectors (OVERWEIGHT)
  - Go short / avoid bottom-ranked sub-sectors (UNDERWEIGHT)
  - Rebalance every 21 trading days (monthly)
  - 10% trailing stop per position
  - Capital: $645 paper (Robinhood, zero equity commission)

Position sizing:
  - Equal-weight across TOPK positions
  - Max 5 positions
  - Use ETF proxies (SMH, IBB, XLE, etc.) for tradability

Reads from:
  - state/subsector_rotation_predictions.json (ML predictions)
  - state/subsector_rotation_state.json (tracker rotation scores)

Outputs:
  - state/subsector_rotation_paper_state.json

Usage:
    python3 paper_engines/subsector_rotation_paper_engine.py
"""

import json
import logging
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)

STATE_PATH = STATE_DIR / "subsector_rotation_paper_state.json"
PREDICTIONS_PATH = STATE_DIR / "subsector_rotation_predictions.json"
TRACKER_STATE_PATH = STATE_DIR / "subsector_rotation_state.json"
TRADE_LOG_PATH = STATE_DIR / "subsector_rotation_paper_trades.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SubSectorPaper] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "subsector_rotation_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy Constants
# ---------------------------------------------------------------------------
CAPITAL_INITIAL = 645.0
MAX_POSITIONS = 5
REBALANCE_DAYS = 21
TRAILING_STOP_PCT = 0.10    # 10% trailing stop
TAKE_PROFIT_PCT = 0.20      # 20% take profit
MAX_HOLD_DAYS = 30           # Max 30 calendar days

# Sub-sector to tradable ETF mapping
SUBSECTOR_ETF_MAP = {
    "semiconductors": "SMH",
    "semicon_equipment": "SMH",
    "enterprise_software": "IGV",
    "cybersecurity": "CIBR",
    "it_hardware": "XLK",
    "biotech": "IBB",
    "medtech_devices": "IHI",
    "pharma_large": "XLV",
    "health_services": "XLV",
    "megabank": "XLF",
    "regional_bank": "KRE",
    "insurance": "KIE",
    "fintech_payments": "IPAY",
    "oil_integrated": "XLE",
    "oilfield_services": "OIH",
    "midstream_pipelines": "AMLP",
    "aerospace_defense": "ITA",
    "heavy_equipment": "XLI",
    "industrial_automation": "XLI",
    "transportation": "IYT",
    "ecommerce_retail": "XLY",
    "autos_ev": "CARZ",
    "restaurants_leisure": "PEJ",
    "consumer_staples_food": "XLP",
    "staples_retail": "XLP",
    "mining_metals": "XME",
    "chemicals": "XLB",
    "reits_data_towers": "XLRE",
    "utilities_electric": "XLU",
    "big_tech_comm": "XLC",
    "telecom": "XLC",
}

# Some sub-sectors map to the same ETF, so we can pick the leader ticker instead
# When two sub-sectors map to the same ETF, prefer the one with better signal
SUBSECTOR_LEADER_TICKER = {
    "semiconductors": "SMH",
    "semicon_equipment": "AMAT",
    "enterprise_software": "IGV",
    "cybersecurity": "CRWD",
    "it_hardware": "AAPL",
    "biotech": "IBB",
    "medtech_devices": "IHI",
    "pharma_large": "LLY",
    "health_services": "UNH",
    "megabank": "JPM",
    "regional_bank": "KRE",
    "insurance": "PGR",
    "fintech_payments": "V",
    "oil_integrated": "XLE",
    "oilfield_services": "OIH",
    "midstream_pipelines": "AMLP",
    "aerospace_defense": "ITA",
    "heavy_equipment": "CAT",
    "industrial_automation": "ETN",
    "transportation": "IYT",
    "ecommerce_retail": "AMZN",
    "autos_ev": "TSLA",
    "restaurants_leisure": "MCD",
    "consumer_staples_food": "PG",
    "staples_retail": "WMT",
    "mining_metals": "XME",
    "chemicals": "LIN",
    "reits_data_towers": "PLD",
    "utilities_electric": "NEE",
    "big_tech_comm": "META",
    "telecom": "T",
}


def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_current_price(ticker: str) -> float:
    """Get current price for a ticker."""
    if not HAS_YF:
        return 0
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0


def load_state() -> dict:
    """Load or initialize paper trading state."""
    state = load_json(STATE_PATH)
    if state is None:
        state = {
            "equity": CAPITAL_INITIAL,
            "cash": CAPITAL_INITIAL,
            "open_positions": [],
            "closed_positions": [],
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0,
            "last_rebalance": None,
            "created_at": datetime.now().isoformat(),
        }
    return state


def save_state(state: dict):
    """Save paper trading state."""
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade: dict):
    """Append trade to JSONL log."""
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def check_exits(state: dict) -> list:
    """Check existing positions for exit conditions."""
    exits = []
    now = datetime.now()

    for pos in state["open_positions"]:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        shares = pos["shares"]
        current_price = get_current_price(ticker)

        if current_price <= 0:
            log.warning(f"  Could not get price for {ticker}, skipping exit check")
            continue

        pos["current_price"] = round(current_price, 2)
        pnl_pct = (current_price - entry_price) / entry_price
        pos["unrealized_pnl_pct"] = round(pnl_pct * 100, 2)

        # Track high water mark
        hwm = pos.get("high_water_mark", entry_price)
        if current_price > hwm:
            pos["high_water_mark"] = current_price
            hwm = current_price

        drawdown_from_peak = (hwm - current_price) / hwm if hwm > 0 else 0

        # Check exit conditions
        exit_reason = None

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = f"TAKE_PROFIT ({pnl_pct*100:.1f}%)"

        # Trailing stop
        elif drawdown_from_peak >= TRAILING_STOP_PCT:
            exit_reason = f"TRAILING_STOP (peak={hwm:.2f}, dd={drawdown_from_peak*100:.1f}%)"

        # Max hold time
        elif pos.get("entry_date"):
            entry_date = datetime.fromisoformat(pos["entry_date"])
            days_held = (now - entry_date).days
            if days_held >= MAX_HOLD_DAYS:
                exit_reason = f"MAX_HOLD ({days_held} days)"

        if exit_reason:
            pnl_dollars = (current_price - entry_price) * shares
            exits.append({
                "ticker": ticker,
                "subsector": pos.get("subsector", ""),
                "entry_price": entry_price,
                "exit_price": current_price,
                "shares": shares,
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "exit_reason": exit_reason,
                "entry_date": pos.get("entry_date"),
                "exit_date": now.isoformat(),
            })

    return exits


def process_exits(state: dict, exits: list):
    """Process exits and update state."""
    for exit_info in exits:
        ticker = exit_info["ticker"]

        # Remove from open positions
        state["open_positions"] = [
            p for p in state["open_positions"] if p["ticker"] != ticker
        ]

        # Add to closed
        state["closed_positions"].append(exit_info)
        state["trade_count"] += 1

        pnl = exit_info["pnl_dollars"]
        state["total_pnl"] += pnl
        state["cash"] += exit_info["exit_price"] * exit_info["shares"]

        if pnl > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1

        log_trade({"action": "EXIT", **exit_info})
        log.info(
            f"  EXIT {ticker} ({exit_info['subsector']}): "
            f"{exit_info['pnl_pct']:+.1f}% (${pnl:+.2f}) — {exit_info['exit_reason']}"
        )


def select_new_positions(state: dict) -> list:
    """Select new positions based on ML predictions and rotation signals."""
    predictions = load_json(PREDICTIONS_PATH)
    tracker_state = load_json(TRACKER_STATE_PATH)

    if not predictions and not tracker_state:
        log.warning("  No predictions or tracker state available")
        return []

    # Build candidate list
    candidates = []

    if predictions and "top_overweight" in predictions:
        for pred in predictions["top_overweight"]:
            subsector = pred["subsector"]
            etf = SUBSECTOR_ETF_MAP.get(subsector)
            leader = SUBSECTOR_LEADER_TICKER.get(subsector)
            tradable = etf or leader

            if not tradable:
                continue

            # Check for conflicting rotation signal from tracker
            rotation_score = 0
            rotation_phase = "UNKNOWN"
            if tracker_state and "subsectors" in tracker_state:
                sub_data = tracker_state["subsectors"].get(subsector, {})
                rotation_score = sub_data.get("rotation_score", 0)
                rotation_phase = sub_data.get("rotation_phase", "UNKNOWN")

            # Composite score: ML prediction + rotation signal
            ml_pred = pred.get("predicted_return_pct", 0)
            composite = ml_pred * 0.6 + rotation_score * 10 * 0.4  # Weight ML more

            # Skip if rotation signal is strongly negative
            if rotation_phase in ("OUTFLOW", "DISTRIBUTING") and ml_pred < 1.0:
                log.info(f"  Skipping {subsector}: rotation {rotation_phase} conflicts with weak ML signal")
                continue

            candidates.append({
                "subsector": subsector,
                "ticker": tradable,
                "ml_predicted_return": ml_pred,
                "rotation_score": rotation_score,
                "rotation_phase": rotation_phase,
                "composite_score": round(composite, 2),
                "rank": pred.get("rank", 99),
            })

    # Also add strong rotation signals even without ML
    if tracker_state and "subsectors" in tracker_state:
        for name, sub_data in tracker_state["subsectors"].items():
            if sub_data.get("rotation_phase") == "INFLOW" and sub_data.get("rotation_score", 0) > 2.0:
                # Check if already a candidate
                if not any(c["subsector"] == name for c in candidates):
                    etf = SUBSECTOR_ETF_MAP.get(name)
                    leader = SUBSECTOR_LEADER_TICKER.get(name)
                    tradable = etf or leader
                    if tradable:
                        candidates.append({
                            "subsector": name,
                            "ticker": tradable,
                            "ml_predicted_return": 0,
                            "rotation_score": sub_data["rotation_score"],
                            "rotation_phase": "INFLOW",
                            "composite_score": sub_data["rotation_score"] * 4,
                            "rank": 99,
                        })

    # Sort by composite score
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    # Avoid duplicate tickers (if two sub-sectors map to same ETF)
    seen_tickers = set(p["ticker"] for p in state["open_positions"])
    filtered = []
    for c in candidates:
        if c["ticker"] not in seen_tickers:
            filtered.append(c)
            seen_tickers.add(c["ticker"])

    # Limit to available slots
    available_slots = MAX_POSITIONS - len(state["open_positions"])
    return filtered[:available_slots]


def process_entries(state: dict, new_positions: list):
    """Open new positions."""
    if not new_positions:
        return

    available_cash = state["cash"]
    position_size = available_cash / len(new_positions) if new_positions else 0

    for pos_info in new_positions:
        ticker = pos_info["ticker"]
        price = get_current_price(ticker)

        if price <= 0:
            log.warning(f"  Could not get price for {ticker}, skipping entry")
            continue

        # Use fractional shares (Robinhood supports this)
        shares = round(position_size / price, 4)
        if shares <= 0.0001:
            log.warning(f"  Insufficient cash for {ticker} @ ${price:.2f}")
            continue

        cost = round(shares * price, 2)
        if cost > available_cash:
            shares = round(available_cash / price, 4)
            cost = round(shares * price, 2)

        if shares <= 0.0001:
            continue

        entry = {
            "ticker": ticker,
            "subsector": pos_info["subsector"],
            "entry_price": round(price, 2),
            "shares": shares,
            "cost": round(cost, 2),
            "entry_date": datetime.now().isoformat(),
            "high_water_mark": round(price, 2),
            "ml_predicted_return": pos_info.get("ml_predicted_return", 0),
            "rotation_score": pos_info.get("rotation_score", 0),
            "rotation_phase": pos_info.get("rotation_phase", ""),
            "composite_score": pos_info.get("composite_score", 0),
        }

        state["open_positions"].append(entry)
        state["cash"] -= cost
        available_cash -= cost

        log_trade({"action": "ENTRY", **entry})
        log.info(
            f"  ENTRY {ticker} ({pos_info['subsector']}): "
            f"{shares} shares @ ${price:.2f} = ${cost:.2f} | "
            f"ML={pos_info.get('ml_predicted_return', 0):+.1f}%, "
            f"rotation={pos_info.get('rotation_score', 0):+.2f}"
        )


def needs_rebalance(state: dict) -> bool:
    """Check if we need to rebalance."""
    last_reb = state.get("last_rebalance")
    if last_reb is None:
        return True

    try:
        last_date = datetime.fromisoformat(last_reb)
        days_since = (datetime.now() - last_date).days
        return days_since >= REBALANCE_DAYS
    except Exception:
        return True


def calculate_portfolio_value(state: dict) -> float:
    """Calculate total portfolio value."""
    pos_value = 0
    for pos in state["open_positions"]:
        price = get_current_price(pos["ticker"])
        if price > 0:
            pos_value += price * pos["shares"]
            pos["current_price"] = round(price, 2)
        else:
            pos_value += pos["entry_price"] * pos["shares"]
    return state["cash"] + pos_value


def run():
    """Main paper trading loop."""
    log.info("=" * 60)
    log.info(f"Sub-Sector Rotation Paper Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Check if it's a weekday
    now = datetime.now()
    if now.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    state = load_state()
    log.info(f"  Equity: ${state['equity']:.2f}, Cash: ${state['cash']:.2f}, "
             f"Positions: {len(state['open_positions'])}")

    # 1. Check exits on existing positions
    if state["open_positions"]:
        exits = check_exits(state)
        if exits:
            process_exits(state, exits)
        else:
            log.info("  No exit signals triggered")

    # 2. Check if rebalance needed
    if needs_rebalance(state):
        log.info("  Rebalance triggered")

        # Exit all remaining positions for rebalance
        if state["open_positions"]:
            all_exits = []
            for pos in state["open_positions"]:
                price = get_current_price(pos["ticker"])
                if price > 0:
                    pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                    pnl_dollars = (price - pos["entry_price"]) * pos["shares"]
                    all_exits.append({
                        "ticker": pos["ticker"],
                        "subsector": pos.get("subsector", ""),
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "shares": pos["shares"],
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "pnl_dollars": round(pnl_dollars, 2),
                        "exit_reason": "REBALANCE",
                        "entry_date": pos.get("entry_date"),
                        "exit_date": now.isoformat(),
                    })
            if all_exits:
                process_exits(state, all_exits)

        # Select and enter new positions
        new_positions = select_new_positions(state)
        if new_positions:
            log.info(f"  Entering {len(new_positions)} new positions")
            process_entries(state, new_positions)
        else:
            log.info("  No new positions to enter")

        state["last_rebalance"] = now.isoformat()
    else:
        # Between rebalances: only check for entries if we have empty slots
        open_slots = MAX_POSITIONS - len(state["open_positions"])
        if open_slots > 0:
            new_positions = select_new_positions(state)
            if new_positions:
                log.info(f"  Filling {len(new_positions)} empty slot(s)")
                process_entries(state, new_positions)

    # Update equity
    state["equity"] = round(calculate_portfolio_value(state), 2)

    # Performance metrics
    total_trades = state["wins"] + state["losses"]
    win_rate = state["wins"] / total_trades * 100 if total_trades > 0 else 0

    log.info(f"\n  Portfolio: ${state['equity']:.2f} "
             f"(P&L: ${state['total_pnl']:+.2f}, {total_trades} trades, WR: {win_rate:.0f}%)")

    # Position details
    if state["open_positions"]:
        log.info("  Open positions:")
        for pos in state["open_positions"]:
            pnl = pos.get("unrealized_pnl_pct", 0)
            log.info(f"    {pos['ticker']} ({pos.get('subsector', '')}): "
                     f"{pos['shares']} sh @ ${pos['entry_price']:.2f} | "
                     f"P&L: {pnl:+.1f}%")

    save_state(state)
    log.info(f"  State saved. {len(state['open_positions'])} open positions.")

    return state


if __name__ == "__main__":
    run()
