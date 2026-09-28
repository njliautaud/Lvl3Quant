#!/usr/bin/env python3
"""
Weekly Momentum Burst Options Paper Engine
============================================

Small-account ($681) strategy buying cheap bull call spreads on momentum gaps.

STRATEGY:
  - Daily: scan for stocks that gapped up 3%+ on above-average volume
  - Buy a bull call spread 1-2 strikes OTM for this week's or next week's expiry
  - Defined risk: spread cost $30-80 (premium paid)
  - Target: 100-200% return on continuation momentum
  - Stop: exit at -50% of premium paid
  - Take profit: exit at +150% of premium paid (2.5x)
  - Time stop: close 1 day before expiry if still open
  - Max concurrent: 4 positions (small bets, diversified)
  - Max position: $80 per spread

PRICING:
  - Paper mode: Black-Scholes estimates flagged as [BS-EST]
  - Commission: $0 (Robinhood)

CAPITAL: $681 starting
MAX POSITION: $80 per spread

Usage:
  python3 paper_engines/weekly_momentum_burst_paper.py
"""

import json
import logging
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state" / "weekly_momentum_burst_state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MOM-BURST] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weekly_momentum_burst.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("momentum_burst")

# ── Config ──
INITIAL_CAPITAL = 681.0
MAX_POSITION_COST = 80.0    # max premium per spread
MAX_CONCURRENT = 4
MIN_GAP_PCT = 3.0            # minimum gap-up %
MIN_VOLUME_MULT = 1.5        # volume must be 1.5x 20-day avg
STRIKES_OTM = 1              # 1-2 strikes OTM for the long call
SPREAD_WIDTH_DOLLARS = 2.5   # $2.50 wide spread (or $5 for higher-priced stocks)
TP_MULT = 2.5                # take profit at 2.5x cost (150% gain)
SL_MULT = 0.5                # stop loss at 50% of cost
RISK_FREE_RATE = 0.05
COMMISSION = 0.0

# Universe: liquid mid/large caps that gap frequently
UNIVERSE = [
    # Tech / growth — gap-prone
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
    "CRM", "NFLX", "ADBE", "ORCL", "SHOP", "PLTR", "SNAP",
    "UBER", "ABNB", "COIN", "RBLX", "NET", "DDOG", "SNOW", "MDB",
    "ZS", "CRWD", "PANW", "FTNT",
    # Biotech / pharma — gap on trial results
    "MRNA", "BNTX", "REGN", "VRTX", "BIIB", "GILD",
    # Consumer / retail — gap on earnings
    "NKE", "LULU", "TGT", "COST", "WMT", "SBUX", "CMG",
    # Energy — gap on macro
    "XLE", "XOM", "CVX", "OXY", "DVN",
    # Financials
    "JPM", "GS", "MS", "BAC",
    # ETFs for sector gaps
    "QQQ", "XLK", "XLF", "XLE", "XLU", "XBI", "ARKK", "SMH",
]

yf = None


def _yf():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf
    return yf


# ────────────────────────────────────────────
#  BLACK-SCHOLES
# ────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ────────────────────────────────────────────
#  STATE
# ────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "positions": [],
        "closed_trades": [],
        "created": str(datetime.now()),
        "last_run": None,
        "total_pnl": 0.0,
        "strategy": "Weekly Momentum Burst - Bull Call Spreads ($681 account)",
    }


def save_state(state):
    state["last_run"] = str(datetime.now())
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(STATE_FILE)


# ────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────
def next_friday(from_date=None):
    """Next Friday expiry."""
    if from_date is None:
        from_date = datetime.now()
    days_until = (4 - from_date.weekday()) % 7
    if days_until < 2:  # need at least 2 days
        days_until += 7
    return from_date + timedelta(days=days_until)


def round_strike(price, tick=1.0):
    """Round to nearest strike increment."""
    if price > 200:
        tick = 5.0
    elif price > 50:
        tick = 2.5
    elif price > 20:
        tick = 1.0
    else:
        tick = 0.5
    return round(price / tick) * tick


def get_stock_data(ticker, days=30):
    try:
        df = _yf().download(ticker, period=f"{days}d", progress=False)
        if df is None or df.empty or len(df) < 5:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        log.warning(f"Data fetch failed for {ticker}: {e}")
        return None


# ────────────────────────────────────────────
#  SCANNING
# ────────────────────────────────────────────
def scan_for_gap_ups(state):
    """Find stocks that gapped up 3%+ today on high volume."""
    today = datetime.now()
    setups = []

    if len(state["positions"]) >= MAX_CONCURRENT:
        log.info(f"At max concurrent ({MAX_CONCURRENT}), skipping scan")
        return []

    capital_in_use = sum(p.get("cost", 0) for p in state["positions"])
    available = state["capital"] - capital_in_use

    log.info(f"Scanning {len(UNIVERSE)} stocks for 3%+ gap-ups (avail: ${available:.2f})")

    for ticker in UNIVERSE:
        try:
            df = get_stock_data(ticker, days=30)
            if df is None or len(df) < 22:
                continue

            # Check today's gap
            current_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            current_open = float(df["Open"].iloc[-1])
            current_vol = float(df["Volume"].iloc[-1])

            # Gap = open vs previous close
            gap_pct = (current_open - prev_close) / prev_close * 100

            if gap_pct < MIN_GAP_PCT:
                continue

            # Volume check: must be above 1.5x 20-day average
            avg_vol = float(df["Volume"].tail(21).iloc[:-1].mean())
            if avg_vol <= 0:
                continue
            vol_mult = current_vol / avg_vol

            if vol_mult < MIN_VOLUME_MULT:
                log.debug(f"  {ticker}: gap {gap_pct:.1f}% but vol only {vol_mult:.1f}x avg")
                continue

            # Skip if we already have a position
            if any(p["ticker"] == ticker for p in state["positions"]):
                continue

            # Compute realized vol for pricing
            returns = df["Close"].pct_change().dropna()
            realized_vol = float(returns.tail(20).std() * np.sqrt(252))
            if realized_vol < 0.05:
                realized_vol = 0.25  # floor

            # Determine spread strikes
            # Long call: 1 strike OTM from current price
            strike_tick = 5.0 if current_close > 200 else (2.5 if current_close > 50 else 1.0)
            long_call = round_strike(current_close + strike_tick * STRIKES_OTM, strike_tick)

            # Spread width
            if current_close > 100:
                sw = 5.0
            elif current_close > 30:
                sw = 2.5
            else:
                sw = 1.0
            short_call = long_call + sw

            # Expiry: this or next Friday
            expiry = next_friday(today)
            T = max((expiry - today).days / 365.0, 1/365.0)

            # Price the spread (BS estimate)
            # Use slightly elevated vol for momentum (stock is moving)
            trade_iv = realized_vol * 1.2  # momentum stocks have higher IV
            long_price = bs_call(current_close, long_call, T, RISK_FREE_RATE, trade_iv)
            short_price = bs_call(current_close, short_call, T, RISK_FREE_RATE, trade_iv)
            spread_cost = (long_price - short_price) * 100  # per contract

            if spread_cost <= 0:
                continue

            # Filter: cost must be $30-80
            if spread_cost < 20:
                log.debug(f"  {ticker}: spread cost ${spread_cost:.0f} too cheap")
                continue
            if spread_cost > MAX_POSITION_COST:
                log.debug(f"  {ticker}: spread cost ${spread_cost:.0f} exceeds max ${MAX_POSITION_COST}")
                continue

            if spread_cost > available:
                continue

            # Max profit = spread width * 100 - cost
            max_profit = sw * 100 - spread_cost

            setup = {
                "ticker": ticker,
                "gap_pct": round(gap_pct, 2),
                "volume_mult": round(vol_mult, 2),
                "current_price": round(current_close, 2),
                "long_call": long_call,
                "short_call": short_call,
                "spread_width": sw,
                "spread_cost": round(spread_cost, 2),
                "max_profit": round(max_profit, 2),
                "tp_target": round(spread_cost * TP_MULT, 2),
                "sl_target": round(spread_cost * SL_MULT, 2),
                "expiry": str(expiry.date()),
                "days_to_expiry": (expiry - today).days,
                "realized_vol": round(realized_vol, 3),
                "trade_iv": round(trade_iv, 3),
                "pricing_method": "[BS-EST]",
            }

            log.info(
                f"  GAP-UP: {ticker} +{gap_pct:.1f}% vol={vol_mult:.1f}x | "
                f"buy {long_call}C/sell {short_call}C | "
                f"cost=${spread_cost:.0f} max_profit=${max_profit:.0f} | "
                f"exp {expiry.date()} [BS-EST]"
            )
            setups.append(setup)

        except Exception as e:
            log.warning(f"  {ticker}: error: {e}")
            continue

    # Sort by gap size * volume (momentum strength)
    setups.sort(key=lambda x: x["gap_pct"] * x["volume_mult"], reverse=True)
    return setups


# ────────────────────────────────────────────
#  POSITION MANAGEMENT
# ────────────────────────────────────────────
def open_positions(state, setups):
    """Open new positions."""
    opened = 0
    for setup in setups:
        if len(state["positions"]) >= MAX_CONCURRENT:
            break

        capital_in_use = sum(p.get("cost", 0) for p in state["positions"])
        available = state["capital"] - capital_in_use
        if setup["spread_cost"] > available:
            continue

        position = {
            "ticker": setup["ticker"],
            "entry_date": str(datetime.now().date()),
            "entry_price": setup["current_price"],
            "long_call": setup["long_call"],
            "short_call": setup["short_call"],
            "spread_width": setup["spread_width"],
            "cost": setup["spread_cost"],
            "max_profit": setup["max_profit"],
            "tp_value": setup["tp_target"],        # close if spread value >= this
            "sl_value": setup["sl_target"],         # close if spread value <= this
            "expiry": setup["expiry"],
            "gap_pct": setup["gap_pct"],
            "volume_mult": setup["volume_mult"],
            "entry_iv": setup["trade_iv"],
            "pricing_method": setup["pricing_method"],
        }

        state["positions"].append(position)
        opened += 1
        log.info(
            f"OPENED: {setup['ticker']} {setup['long_call']}C/{setup['short_call']}C "
            f"cost=${setup['spread_cost']:.0f} TP=${setup['tp_target']:.0f} "
            f"SL=${setup['sl_target']:.0f} exp={setup['expiry']}"
        )

    return opened


def check_exits(state):
    """Check positions for TP/SL/expiry exits."""
    today = datetime.now()
    closed = 0
    remaining = []

    for pos in state["positions"]:
        ticker = pos["ticker"]
        should_close = False
        close_reason = ""
        pnl = 0.0

        df = get_stock_data(ticker, days=10)
        if df is None:
            remaining.append(pos)
            continue

        current_price = float(df["Close"].iloc[-1])
        pos["current_price"] = round(current_price, 2)

        expiry_dt = datetime.strptime(pos["expiry"], "%Y-%m-%d")
        days_to_expiry = (expiry_dt - today).days

        # Compute current spread value
        returns = df["Close"].pct_change().dropna()
        current_vol = float(returns.tail(10).std() * np.sqrt(252)) if len(returns) >= 10 else pos["entry_iv"]
        if current_vol < 0.05:
            current_vol = 0.20
        T = max(days_to_expiry / 365.0, 1/365.0)

        long_val = bs_call(current_price, pos["long_call"], T, RISK_FREE_RATE, current_vol)
        short_val = bs_call(current_price, pos["short_call"], T, RISK_FREE_RATE, current_vol)
        spread_value = (long_val - short_val) * 100

        # At expiry
        if days_to_expiry <= 0:
            should_close = True
            close_reason = "expiry"
            # Intrinsic value at expiry
            long_intrinsic = max(current_price - pos["long_call"], 0)
            short_intrinsic = max(current_price - pos["short_call"], 0)
            final_value = (long_intrinsic - short_intrinsic) * 100
            pnl = final_value - pos["cost"]

        # Time stop: close 1 day before expiry
        elif days_to_expiry <= 1:
            should_close = True
            close_reason = "time stop (1 day before expiry)"
            pnl = spread_value - pos["cost"]

        # Take profit
        elif spread_value >= pos["tp_value"]:
            should_close = True
            close_reason = f"TAKE PROFIT: spread=${spread_value:.0f} >= TP=${pos['tp_value']:.0f}"
            pnl = spread_value - pos["cost"]

        # Stop loss
        elif spread_value <= pos["sl_value"]:
            should_close = True
            close_reason = f"STOP LOSS: spread=${spread_value:.0f} <= SL=${pos['sl_value']:.0f}"
            pnl = spread_value - pos["cost"]

        if should_close:
            trade = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": str(today.date()),
                "long_call": pos["long_call"],
                "short_call": pos["short_call"],
                "cost": pos["cost"],
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / pos["cost"] * 100, 1) if pos["cost"] > 0 else 0,
                "close_reason": close_reason,
                "entry_price": pos["entry_price"],
                "exit_price": current_price,
                "gap_pct": pos["gap_pct"],
                "pricing_method": "[BS-EST]",
            }
            state["closed_trades"].append(trade)
            state["capital"] += pnl
            state["total_pnl"] = state.get("total_pnl", 0) + pnl
            closed += 1
            log.info(
                f"CLOSED: {ticker} {pos['long_call']}C/{pos['short_call']}C | "
                f"P&L=${pnl:.2f} ({trade['return_pct']:.0f}%) | {close_reason}"
            )
        else:
            # Log unrealized P&L
            unrealized = spread_value - pos["cost"]
            log.info(
                f"  HOLD: {ticker} {pos['long_call']}C/{pos['short_call']}C | "
                f"unreal=${unrealized:.0f} spread_val=${spread_value:.0f} | "
                f"{days_to_expiry}d to exp"
            )
            remaining.append(pos)

    state["positions"] = remaining
    return closed


# ────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("WEEKLY MOMENTUM BURST PAPER ENGINE — daily run")
    log.info("=" * 60)

    state = load_state()

    # 1) Check exits
    closed = check_exits(state)
    if closed:
        log.info(f"Closed {closed} positions")

    # 2) Scan for gap-ups
    setups = scan_for_gap_ups(state)
    log.info(f"Found {len(setups)} momentum gap-up setups")

    # 3) Open best setups
    if setups:
        opened = open_positions(state, setups)
        log.info(f"Opened {opened} new positions")

    # 4) Summary
    capital_in_use = sum(p.get("cost", 0) for p in state["positions"])
    total_trades = len(state["closed_trades"])
    winners = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
    wr = winners / total_trades * 100 if total_trades > 0 else 0
    avg_return = (
        np.mean([t["return_pct"] for t in state["closed_trades"]])
        if total_trades > 0 else 0
    )

    log.info(f"--- SUMMARY ---")
    log.info(f"Capital: ${state['capital']:.2f} | In use: ${capital_in_use:.2f}")
    log.info(f"Open: {len(state['positions'])} | Closed: {total_trades}")
    log.info(f"Win rate: {wr:.0f}% | Avg return: {avg_return:.0f}%")
    log.info(f"Total P&L: ${state.get('total_pnl', 0):.2f}")

    save_state(state)
    log.info("Done.")


if __name__ == "__main__":
    run()
