#!/usr/bin/env python3
"""
ETF Short-Term Reversal Paper Trading Engine (Strategy 1B)
==========================================================

Validated strategy from the quantitative strategy catalog.
Passed ALL validation gates:
  - Permutation p-value: 0.000
  - R1 Regime Gap: 0.03 - 0.23 (well under 0.50 threshold)
  - Sub-period consistency: PASSED

Backtest stats (2010-2026):
  - Sharpe Ratio:   0.83
  - Win Rate:       58%
  - Profit Factor:  1.43
  - Annual Return:  14.7%

Strategy Rules (fixed params, NO optimization):
  - Universe: 21 ETFs (11 sector SPDRs + 10 broad/other)
  - Every Monday: rank all 21 by trailing 5-day return
  - Buy the 5 WORST performers with equal weight
  - Hold for 5 trading days (sell next Monday)
  - Commission: $0 (Robinhood)
  - Spread cost: 2 bps per trade (0.02%)

Usage:
  python3 etf_reversal_paper.py           # daily update (rebalance on Monday)
  python3 etf_reversal_paper.py --status  # print current positions & NAV
  python3 etf_reversal_paper.py --history # print last 20 equity curve entries

Designed for PM2 cron, runs weekdays at 10:00 AM ET.

State:  /home/jupiter/Lvl3Quant/state/etf_reversal/
Logs:   /home/jupiter/Lvl3Quant/logs/etf_reversal_paper.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # /home/jupiter/Lvl3Quant
STATE_DIR = ROOT / "state" / "etf_reversal"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
EQUITY_FILE = STATE_DIR / "equity_curve.jsonl"
LOG_FILE = ROOT / "logs" / "etf_reversal_paper.log"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 100_000.0
SPREAD_BPS = 2  # 0.02% per trade
TOP_N = 5  # buy bottom 5 performers
LOOKBACK_DAYS = 5  # trailing 5-day return
HOLD_DAYS = 5  # hold for 5 trading days

UNIVERSE = [
    # 11 Sector SPDRs
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # Broad market / other
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "EEM", "HYG", "UUP",
]
assert len(UNIVERSE) == 21

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("etf_reversal")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def _default_state() -> dict[str, Any]:
    return {
        "positions": {},        # {ticker: {"shares": float, "entry_price": float, "entry_date": str}}
        "cash": STARTING_CAPITAL,
        "nav": STARTING_CAPITAL,
        "last_rebalance_date": None,
        "inception_date": None,
        "trade_count": 0,
    }


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return _default_state()


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(STATE_FILE)


def append_trade(record: dict) -> None:
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def append_equity(record: dict) -> None:
    with open(EQUITY_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_prices(tickers: list[str], days: int = 15) -> dict[str, dict]:
    """Fetch recent prices via yfinance. Returns {ticker: {"prices": [...], "current": float}}."""
    import yfinance as yf

    end = datetime.now(ET)
    # Fetch extra calendar days to ensure we get enough trading days
    start = end - timedelta(days=days + 10)

    logger.debug(f"Fetching prices for {len(tickers)} tickers from {start.date()} to {end.date()}")

    result = {}
    try:
        data = yf.download(
            tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return {}

    if data.empty:
        logger.error("yfinance returned empty dataframe")
        return {}

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                close = data["Close"].dropna()
            else:
                close = data["Close"][ticker].dropna()

            if len(close) < 2:
                logger.warning(f"{ticker}: insufficient data ({len(close)} rows)")
                continue

            prices = close.values.tolist()
            result[ticker] = {
                "prices": prices,
                "current": prices[-1],
                "dates": [d.strftime("%Y-%m-%d") for d in close.index],
            }
        except Exception as e:
            logger.warning(f"{ticker}: error extracting data: {e}")
            continue

    logger.debug(f"Got data for {len(result)}/{len(tickers)} tickers")
    return result


def compute_trailing_returns(price_data: dict[str, dict], lookback: int = LOOKBACK_DAYS) -> dict[str, float]:
    """Compute trailing N-day returns for each ticker."""
    returns = {}
    for ticker, data in price_data.items():
        prices = data["prices"]
        if len(prices) < lookback + 1:
            logger.warning(f"{ticker}: only {len(prices)} prices, need {lookback + 1}")
            continue
        ret = (prices[-1] / prices[-(lookback + 1)]) - 1.0
        returns[ticker] = ret
    return returns


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------
def apply_spread_cost(price: float, side: str) -> float:
    """Apply spread cost (2 bps)."""
    if side == "buy":
        return price * (1 + SPREAD_BPS / 10_000)
    else:  # sell
        return price * (1 - SPREAD_BPS / 10_000)


def execute_sells(state: dict, price_data: dict[str, dict], today_str: str) -> float:
    """Sell all current positions. Returns total proceeds."""
    total_proceeds = 0.0
    positions = dict(state["positions"])  # copy to avoid mutation during iteration

    for ticker, pos in positions.items():
        if ticker not in price_data:
            logger.error(f"Cannot sell {ticker}: no price data. Keeping position.")
            continue

        current_price = price_data[ticker]["current"]
        sell_price = apply_spread_cost(current_price, "sell")
        proceeds = pos["shares"] * sell_price
        total_proceeds += proceeds

        pnl = (sell_price - pos["entry_price"]) * pos["shares"]
        pnl_pct = (sell_price / pos["entry_price"] - 1) * 100

        trade_record = {
            "date": today_str,
            "ticker": ticker,
            "side": "SELL",
            "shares": pos["shares"],
            "price": round(sell_price, 4),
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
        }
        append_trade(trade_record)
        logger.info(f"SELL {pos['shares']:.2f} {ticker} @ ${sell_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

        del state["positions"][ticker]

    state["cash"] += total_proceeds
    return total_proceeds


def execute_buys(state: dict, bottom_5: list[str], price_data: dict[str, dict], today_str: str) -> None:
    """Buy equal-weight positions in the bottom 5 ETFs."""
    available_cash = state["cash"]
    per_position = available_cash / len(bottom_5)

    for ticker in bottom_5:
        if ticker not in price_data:
            logger.error(f"Cannot buy {ticker}: no price data. Skipping.")
            continue

        current_price = price_data[ticker]["current"]
        buy_price = apply_spread_cost(current_price, "buy")
        shares = per_position / buy_price

        cost = shares * buy_price
        state["cash"] -= cost

        state["positions"][ticker] = {
            "shares": round(shares, 6),
            "entry_price": round(buy_price, 4),
            "entry_date": today_str,
        }

        trade_record = {
            "date": today_str,
            "ticker": ticker,
            "side": "BUY",
            "shares": round(shares, 6),
            "price": round(buy_price, 4),
            "cost": round(cost, 2),
        }
        append_trade(trade_record)
        logger.info(f"BUY  {shares:.2f} {ticker} @ ${buy_price:.2f} (${cost:.2f})")

    state["trade_count"] += len(bottom_5) * 2  # buy + future sell


def mark_to_market(state: dict, price_data: dict[str, dict]) -> float:
    """Update NAV based on current prices. Returns NAV."""
    portfolio_value = state["cash"]

    for ticker, pos in state["positions"].items():
        if ticker in price_data:
            current_price = price_data[ticker]["current"]
            portfolio_value += pos["shares"] * current_price
        else:
            # Use entry price as fallback
            portfolio_value += pos["shares"] * pos["entry_price"]
            logger.warning(f"{ticker}: no current price, using entry price for MTM")

    state["nav"] = round(portfolio_value, 2)
    return portfolio_value


def is_monday(dt: datetime) -> bool:
    """Check if the given datetime is Monday."""
    return dt.weekday() == 0


# ---------------------------------------------------------------------------
# Main daily logic
# ---------------------------------------------------------------------------
def run_daily() -> None:
    """Execute daily update. Rebalance on Monday, mark-to-market other days."""
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        logger.info(f"{today_str} is a weekend. Nothing to do.")
        return

    state = load_state()

    # Set inception date on first run
    if state["inception_date"] is None:
        state["inception_date"] = today_str
        logger.info(f"Inception date set to {today_str}")

    # Fetch price data for entire universe
    price_data = fetch_prices(UNIVERSE)
    if not price_data:
        logger.error("Failed to fetch any price data. Aborting.")
        save_state(state)
        return

    if len(price_data) < 15:
        logger.warning(f"Only got data for {len(price_data)}/{len(UNIVERSE)} tickers")

    monday = is_monday(now)

    if monday:
        logger.info(f"=== REBALANCE DAY ({today_str}) ===")

        # Check if we already rebalanced today
        if state["last_rebalance_date"] == today_str:
            logger.info("Already rebalanced today. Running mark-to-market only.")
        else:
            # Step 1: Sell all existing positions
            if state["positions"]:
                logger.info(f"Selling {len(state['positions'])} positions...")
                execute_sells(state, price_data, today_str)

            # Step 2: Rank by trailing 5-day return, pick bottom 5
            returns = compute_trailing_returns(price_data, LOOKBACK_DAYS)
            if len(returns) < TOP_N:
                logger.error(f"Only {len(returns)} tickers with valid returns. Need at least {TOP_N}. Skipping rebalance.")
                save_state(state)
                return

            sorted_returns = sorted(returns.items(), key=lambda x: x[1])
            bottom_5 = [t for t, _ in sorted_returns[:TOP_N]]
            bottom_5_returns = {t: r for t, r in sorted_returns[:TOP_N]}

            logger.info("Bottom 5 ETFs by 5-day return:")
            for ticker, ret in sorted_returns[:TOP_N]:
                logger.info(f"  {ticker}: {ret*100:+.2f}%")

            # Step 3: Buy bottom 5 equal weight
            execute_buys(state, bottom_5, price_data, today_str)
            state["last_rebalance_date"] = today_str

    else:
        logger.info(f"=== MARK-TO-MARKET ({today_str}) ===")

    # Mark to market
    nav = mark_to_market(state, price_data)

    # Calculate return since inception
    inception_return = (nav / STARTING_CAPITAL - 1) * 100

    logger.info(f"NAV: ${nav:,.2f} | Return: {inception_return:+.2f}% | Cash: ${state['cash']:,.2f} | Positions: {len(state['positions'])}")

    # Log equity curve
    equity_record = {
        "date": today_str,
        "nav": round(nav, 2),
        "cash": round(state["cash"], 2),
        "positions": len(state["positions"]),
        "return_pct": round(inception_return, 4),
    }
    append_equity(equity_record)

    # Save state
    save_state(state)
    logger.info("State saved.")


# ---------------------------------------------------------------------------
# CLI: --status
# ---------------------------------------------------------------------------
def print_status() -> None:
    """Print current positions and NAV."""
    state = load_state()
    print("=" * 60)
    print("ETF Short-Term Reversal Paper Trading Engine")
    print("=" * 60)
    print(f"Inception:      {state.get('inception_date', 'N/A')}")
    print(f"NAV:            ${state['nav']:,.2f}")
    print(f"Cash:           ${state['cash']:,.2f}")
    print(f"Total trades:   {state['trade_count']}")
    print(f"Last rebalance: {state.get('last_rebalance_date', 'N/A')}")
    print()

    if state["positions"]:
        print(f"Open positions ({len(state['positions'])}):")
        print(f"  {'Ticker':<8} {'Shares':>10} {'Entry Price':>12} {'Entry Date':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
        for ticker, pos in sorted(state["positions"].items()):
            print(f"  {ticker:<8} {pos['shares']:>10.2f} ${pos['entry_price']:>10.2f} {pos['entry_date']:>12}")
    else:
        print("No open positions.")

    inception_return = (state["nav"] / STARTING_CAPITAL - 1) * 100
    print(f"\nReturn since inception: {inception_return:+.2f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI: --history
# ---------------------------------------------------------------------------
def print_history(n: int = 20) -> None:
    """Print last N equity curve entries."""
    if not EQUITY_FILE.exists():
        print("No equity curve data yet.")
        return

    lines = []
    with open(EQUITY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))

    if not lines:
        print("No equity curve data yet.")
        return

    recent = lines[-n:]
    print("=" * 60)
    print("ETF Reversal Equity Curve (last {} entries)".format(len(recent)))
    print("=" * 60)
    print(f"  {'Date':<12} {'NAV':>12} {'Return':>10} {'Positions':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for entry in recent:
        print(f"  {entry['date']:<12} ${entry['nav']:>10,.2f} {entry['return_pct']:>+9.2f}% {entry['positions']:>10}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Short-Term Reversal Paper Trading Engine")
    parser.add_argument("--status", action="store_true", help="Print current positions and NAV")
    parser.add_argument("--history", action="store_true", help="Print last 20 equity curve entries")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.history:
        print_history()
    else:
        run_daily()


if __name__ == "__main__":
    main()
