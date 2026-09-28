#!/usr/bin/env python3
"""
VIX Vol Term Structure Paper Trading Engine
============================================

Daily paper engine: buy quality stocks when VIX spikes and starts declining.

STRATEGY:
  - Signal: VIX > 1.15 × VIX_60day_rolling_mean AND VIX today < VIX yesterday
  - Entry: Equal-weight basket of 20 quality stocks ($15 each = $300 total)
  - Max 2 concurrent baskets
  - Exit conditions (any):
    1. VIX drops below 60-day rolling mean
    2. 21-day max hold reached
    3. +10% portfolio gain on the basket
    4. -15% portfolio loss on the basket

Usage:
  python3 paper_engines/vol_term_structure_paper.py

Author: Claude (autonomous build)
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

yf = None


def _import_yfinance():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf


# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "vol_term_structure_state.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "vol_term_structure_paper.log"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [VOL-TERM] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("vol_term_structure")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
BASKET_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
    "NFLX", "ADBE", "PG", "JNJ",
]

BASKET_TOTAL = 300.0          # $300 per basket entry
PER_STOCK = BASKET_TOTAL / len(BASKET_TICKERS)  # $15 each
MAX_CONCURRENT_BASKETS = 2
VIX_SPIKE_MULT = 1.15         # VIX must be > 1.15 × 60d mean
VIX_ROLLING_WINDOW = 60       # 60-day rolling mean
MAX_HOLD_DAYS = 21
TP_PCT = 0.10                 # +10% portfolio gain
SL_PCT = -0.15                # -15% portfolio loss


# ─────────────────────────────────────────────
#  STATE MANAGEMENT
# ─────────────────────────────────────────────
def _default_state() -> Dict[str, Any]:
    return {
        "open_baskets": [],       # list of basket dicts
        "trade_history": [],      # closed baskets
        "total_realized_pnl": 0.0,
        "last_run": None,
        "total_entries": 0,
        "total_exits": 0,
    }


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            log.warning("Corrupt state file, starting fresh")
    return _default_state()


def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────
def get_vix_data() -> Optional[Dict[str, float]]:
    """Fetch VIX current, yesterday, and 60-day rolling mean."""
    _import_yfinance()
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="90d")
        if hist.empty or len(hist) < VIX_ROLLING_WINDOW + 1:
            log.error("Insufficient VIX data (got %d rows, need %d)", len(hist), VIX_ROLLING_WINDOW + 1)
            return None

        vix_today = float(hist["Close"].iloc[-1])
        vix_yesterday = float(hist["Close"].iloc[-2])
        vix_60d_mean = float(hist["Close"].iloc[-VIX_ROLLING_WINDOW:].mean())

        return {
            "vix_today": vix_today,
            "vix_yesterday": vix_yesterday,
            "vix_60d_mean": vix_60d_mean,
            "date": str(hist.index[-1].date()),
        }
    except Exception as e:
        log.error("Failed to fetch VIX data: %s", e)
        return None


def get_stock_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of tickers."""
    _import_yfinance()
    prices = {}
    try:
        data = yf.download(tickers, period="1d", progress=False, threads=True)
        if data.empty:
            log.error("No stock price data returned")
            return prices
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    price = float(data["Close"].iloc[-1])
                else:
                    price = float(data["Close"][ticker].iloc[-1])
                if not np.isnan(price) and price > 0:
                    prices[ticker] = price
            except (KeyError, IndexError):
                log.warning("No price for %s", ticker)
    except Exception as e:
        log.error("Failed to download stock prices: %s", e)
    return prices


# ─────────────────────────────────────────────
#  ENTRY LOGIC
# ─────────────────────────────────────────────
def check_entry_signal(vix_data: Dict[str, float]) -> bool:
    """
    Entry signal fires when:
    1. VIX > 1.15 × 60-day rolling mean (fear is elevated)
    2. VIX today < VIX yesterday (fear peaked, now declining)
    """
    vix_today = vix_data["vix_today"]
    vix_yesterday = vix_data["vix_yesterday"]
    vix_60d_mean = vix_data["vix_60d_mean"]
    threshold = VIX_SPIKE_MULT * vix_60d_mean

    is_elevated = vix_today > threshold
    is_declining = vix_today < vix_yesterday

    log.info("VIX=%.2f, Yesterday=%.2f, 60d_mean=%.2f, threshold=%.2f (1.15×mean)",
             vix_today, vix_yesterday, vix_60d_mean, threshold)
    log.info("Elevated: %s, Declining: %s → Signal: %s",
             is_elevated, is_declining, is_elevated and is_declining)

    return is_elevated and is_declining


def enter_basket(state: Dict[str, Any], vix_data: Dict[str, float]) -> bool:
    """Enter a new basket position."""
    if len(state["open_baskets"]) >= MAX_CONCURRENT_BASKETS:
        log.info("Max concurrent baskets (%d) reached, skipping entry", MAX_CONCURRENT_BASKETS)
        return False

    prices = get_stock_prices(BASKET_TICKERS)
    if len(prices) < 15:
        log.error("Too few prices fetched (%d/%d), skipping entry", len(prices), len(BASKET_TICKERS))
        return False

    # Build position: $15 per stock
    positions = {}
    total_cost = 0.0
    for ticker in BASKET_TICKERS:
        if ticker in prices:
            price = prices[ticker]
            shares = PER_STOCK / price  # fractional shares OK for paper
            positions[ticker] = {
                "entry_price": round(price, 4),
                "shares": round(shares, 6),
                "cost": round(shares * price, 2),
            }
            total_cost += shares * price

    basket = {
        "id": state["total_entries"] + 1,
        "entry_date": vix_data["date"],
        "entry_vix": vix_data["vix_today"],
        "entry_vix_60d_mean": vix_data["vix_60d_mean"],
        "positions": positions,
        "total_cost": round(total_cost, 2),
        "days_held": 0,
    }

    state["open_baskets"].append(basket)
    state["total_entries"] += 1

    log.info("ENTRY basket #%d: %d stocks, $%.2f invested, VIX=%.2f",
             basket["id"], len(positions), total_cost, vix_data["vix_today"])
    return True


# ─────────────────────────────────────────────
#  EXIT LOGIC
# ─────────────────────────────────────────────
def check_exits(state: Dict[str, Any], vix_data: Dict[str, float]):
    """Check all open baskets for exit conditions."""
    if not state["open_baskets"]:
        return

    # Fetch current prices for all tickers in open baskets
    all_tickers = set()
    for basket in state["open_baskets"]:
        all_tickers.update(basket["positions"].keys())
    current_prices = get_stock_prices(list(all_tickers))

    if not current_prices:
        log.warning("No current prices available, skipping exit check")
        return

    baskets_to_close = []
    for basket in state["open_baskets"]:
        # Calculate current basket value and P&L
        current_value = 0.0
        for ticker, pos in basket["positions"].items():
            if ticker in current_prices:
                current_value += pos["shares"] * current_prices[ticker]
            else:
                # Use entry price as fallback
                current_value += pos["shares"] * pos["entry_price"]

        pnl = current_value - basket["total_cost"]
        pnl_pct = pnl / basket["total_cost"] if basket["total_cost"] > 0 else 0.0

        # Calculate days held
        entry_date = datetime.strptime(basket["entry_date"], "%Y-%m-%d")
        today = datetime.strptime(vix_data["date"], "%Y-%m-%d")
        days_held = (today - entry_date).days
        basket["days_held"] = days_held

        # Check exit conditions
        exit_reason = None
        vix_today = vix_data["vix_today"]
        vix_60d_mean = vix_data["vix_60d_mean"]

        if vix_today < vix_60d_mean:
            exit_reason = f"VIX normalized ({vix_today:.2f} < 60d_mean {vix_60d_mean:.2f})"
        elif days_held >= MAX_HOLD_DAYS:
            exit_reason = f"Max hold {MAX_HOLD_DAYS} days reached"
        elif pnl_pct >= TP_PCT:
            exit_reason = f"Take profit +{pnl_pct*100:.1f}%"
        elif pnl_pct <= SL_PCT:
            exit_reason = f"Stop loss {pnl_pct*100:.1f}%"

        log.info("Basket #%d: day %d, value=$%.2f, cost=$%.2f, P&L=$%.2f (%.1f%%)",
                 basket["id"], days_held, current_value, basket["total_cost"],
                 pnl, pnl_pct * 100)

        if exit_reason:
            baskets_to_close.append((basket, exit_reason, pnl, pnl_pct, current_value, current_prices))

    # Close baskets
    for basket, reason, pnl, pnl_pct, current_value, prices in baskets_to_close:
        log.info("EXIT basket #%d: %s | P&L=$%.2f (%.1f%%)",
                 basket["id"], reason, pnl, pnl_pct * 100)

        trade_record = {
            "basket_id": basket["id"],
            "entry_date": basket["entry_date"],
            "exit_date": vix_data["date"],
            "entry_vix": basket["entry_vix"],
            "exit_vix": vix_data["vix_today"],
            "days_held": basket["days_held"],
            "total_cost": basket["total_cost"],
            "exit_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "exit_reason": reason,
            "num_stocks": len(basket["positions"]),
        }
        state["trade_history"].append(trade_record)
        state["total_realized_pnl"] = round(state["total_realized_pnl"] + pnl, 2)
        state["total_exits"] += 1
        state["open_baskets"].remove(basket)


# ─────────────────────────────────────────────
#  UNREALIZED P&L
# ─────────────────────────────────────────────
def calc_unrealized_pnl(state: Dict[str, Any]) -> float:
    """Calculate total unrealized P&L across open baskets."""
    if not state["open_baskets"]:
        return 0.0

    all_tickers = set()
    for basket in state["open_baskets"]:
        all_tickers.update(basket["positions"].keys())
    current_prices = get_stock_prices(list(all_tickers))

    total_unrealized = 0.0
    for basket in state["open_baskets"]:
        for ticker, pos in basket["positions"].items():
            current = current_prices.get(ticker, pos["entry_price"])
            total_unrealized += pos["shares"] * (current - pos["entry_price"])

    return round(total_unrealized, 2)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("VIX Vol Term Structure Paper Engine — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    state = load_state()

    # 1. Fetch VIX data
    vix_data = get_vix_data()
    if vix_data is None:
        log.error("Cannot proceed without VIX data")
        return

    # 2. Check exits first
    check_exits(state, vix_data)

    # 3. Check entry signal
    signal_fires = check_entry_signal(vix_data)
    entered = False
    if signal_fires:
        entered = enter_basket(state, vix_data)

    # 4. Calculate unrealized P&L (only if we have open positions and didn't just fetch prices)
    unrealized_pnl = 0.0
    if state["open_baskets"]:
        unrealized_pnl = calc_unrealized_pnl(state)

    # 5. Update state
    state["last_run"] = datetime.now().isoformat()
    save_state(state)

    # 6. Print daily summary
    n_open = len(state["open_baskets"])
    n_trades = len(state["trade_history"])
    realized = state["total_realized_pnl"]
    signal_str = "FIRED" if signal_fires else "no signal"
    entry_str = " (entered)" if entered else ""

    summary = (
        f"[SUMMARY] {vix_data['date']} | VIX={vix_data['vix_today']:.2f} "
        f"(60d_avg={vix_data['vix_60d_mean']:.2f}) | Signal: {signal_str}{entry_str} | "
        f"Open: {n_open} baskets | Unrealized: ${unrealized_pnl:+.2f} | "
        f"Realized: ${realized:+.2f} | Trades: {n_trades}"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    run()
