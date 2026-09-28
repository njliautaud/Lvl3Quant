#!/usr/bin/env python3
"""
Iron Condor Paper Trading Engine
==================================
Paper trades 7-DTE iron condors on liquid mega-cap stocks with VIX gating.

Based on validated backtest config: IC_7d_VIX (Sharpe 4.28, CAGR 52%, DD -7.5%)
Real bid-ask calibration confirms 10% BA assumption matches live Robinhood data
for the target universe (AAPL 12%, MSFT 6%, AMZN 5% → portfolio avg ~8-10%).

Strategy:
  - Sell 0.20-delta put spread + 0.20-delta call spread (iron condor)
  - 7 DTE target, wings 5% of strike OTM from short strikes
  - VIX gating: reduce sizing when VIX is elevated
  - Profit take at 50% of credit received
  - Stop loss at 2x credit received
  - Max 15 concurrent positions, 4% of capital per name
  - Earnings buffer: skip if earnings within 7 days

Runs daily at 15:55 ET via PM2 cron.
State: /home/jupiter/Lvl3Quant/state/ic_paper_state.json
History: /home/jupiter/Lvl3Quant/state/ic_paper_history.csv

PM2 ecosystem entry:
{
  name: "ic-paper-engine",
  script: "/home/jupiter/Lvl3Quant/engines/iron_condor_paper.py",
  interpreter: "python3",
  cron_restart: "55 19 * * 1-5",
  autorestart: false,
  ...
}
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
import traceback
import urllib.request
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────
ET = pytz.timezone("US/Eastern")
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "ic_paper_state.json"
HISTORY_FILE = STATE_DIR / "ic_paper_history.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK_FILE = ROOT.parent / "teleclaude-main" / "API_KEYS.md"

# Strategy parameters (validated config)
INITIAL_CAPITAL = 100_000
DTE_TARGET = 7          # Target 7 DTE
DTE_MIN = 5             # Min acceptable DTE
DTE_MAX = 10            # Max acceptable DTE
PUT_DELTA = -0.20       # Target short put delta
CALL_DELTA = 0.20       # Target short call delta
WING_OFFSET_PCT = 0.05  # Long leg is 5% of strike further OTM
PROFIT_TAKE = 0.50      # Close at 50% of max profit
STOP_LOSS_MULT = 2.0    # Close at 2x credit lost
MAX_CONCURRENT = 15     # Max open ICs
PER_NAME_PCT = 0.04     # Max 4% of portfolio per name
EARNINGS_BUFFER = 7     # Skip if earnings within 7 calendar days

# Universe: liquid mega-caps with tight option spreads
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "JNJ", "MA", "XOM",
    "COST", "ABBV", "KO", "PEP", "MRK",
]

# VIX gating: scale down when VIX is elevated
VIX_TIERS = [
    (30, 0.00),   # VIX > 30: no new positions
    (25, 0.25),   # VIX 25-30: 25% sizing
    (20, 0.50),   # VIX 20-25: 50% sizing
    (15, 0.80),   # VIX 15-20: 80% sizing
    (0,  1.00),   # VIX < 15: full sizing
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IC-Paper] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ic_paper_engine.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ic_paper")

# ── Black-Scholes ─────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        if option_type == "put":
            return max(K - S, 0)
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def find_strike_by_delta(S, strikes, T, r, sigma, target_delta, option_type="put"):
    """Find the strike closest to target delta."""
    best_strike = None
    best_diff = float("inf")
    for K in strikes:
        d = bs_delta(S, K, T, r, sigma, option_type)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = K
    return best_strike


# ── VIX gating ────────────────────────────────────────────────────────────

def vix_sizing_mult(vix_level: float) -> float:
    """Return position sizing multiplier based on VIX level."""
    for threshold, mult in VIX_TIERS:
        if vix_level > threshold:
            return mult
    return 1.0


# ── Earnings calendar ─────────────────────────────────────────────────────

def has_earnings_soon(ticker: str, within_days: int = EARNINGS_BUFFER) -> bool:
    """Check if ticker has earnings within N calendar days."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and not cal.empty:
            if isinstance(cal, pd.DataFrame):
                # yfinance returns different formats
                if "Earnings Date" in cal.index:
                    dates = cal.loc["Earnings Date"]
                    for d in dates:
                        if isinstance(d, (pd.Timestamp, datetime)):
                            days_to = (d.date() - date.today()).days
                            if 0 <= days_to <= within_days:
                                return True
        # Also check from earnings_dates
        edates = t.earnings_dates
        if edates is not None and len(edates) > 0:
            for d in edates.index:
                if isinstance(d, (pd.Timestamp, datetime)):
                    days_to = (d.date() - date.today()).days
                    if -2 <= days_to <= within_days:
                        return True
    except Exception:
        pass
    return False


# ── State management ──────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "next_id": 1,
        "start_date": datetime.now(ET).strftime("%Y-%m-%d"),
        "total_trades": 0,
        "total_wins": 0,
        "total_pnl": 0.0,
        "last_update": None,
    }


def save_state(state: dict):
    state["last_update"] = datetime.now(ET).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def append_history(row: dict):
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


# ── Discord alerting ──────────────────────────────────────────────────────

def get_webhook_url() -> Optional[str]:
    try:
        text = WEBHOOK_FILE.read_text()
        for line in text.splitlines():
            if "discord" in line.lower() and "webhook" in line.lower():
                import re
                urls = re.findall(r'https://discord\.com/api/webhooks/\S+', line)
                if urls:
                    return urls[0].rstrip('`').rstrip("'").rstrip('"')
    except Exception:
        pass
    return None


def send_alert(message: str):
    log.info(f"ALERT: {message}")
    url = get_webhook_url()
    if not url:
        return
    try:
        payload = json.dumps({"content": message[:1990]}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning(f"Failed to send alert: {e}")


# ── Core logic ────────────────────────────────────────────────────────────

def get_vix() -> float:
    """Get current VIX level."""
    try:
        vix = yf.download("^VIX", period="2d", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return float(vix["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"Failed to get VIX: {e}")
        return 20.0  # Default to neutral


def get_risk_free_rate() -> float:
    """Get approximate risk-free rate from 13-week T-bill."""
    try:
        irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        if isinstance(irx.columns, pd.MultiIndex):
            irx.columns = irx.columns.get_level_values(0)
        return float(irx["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.05  # Default 5%


def scan_for_ic(ticker: str, S: float, vix: float, r: float) -> Optional[dict]:
    """
    Scan a ticker for an iron condor opportunity.
    Returns dict with trade details or None.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None

        # Find expiration closest to DTE_TARGET
        today = date.today()
        best_exp = None
        best_dte_diff = float("inf")

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                diff = abs(dte - DTE_TARGET)
                if diff < best_dte_diff:
                    best_dte_diff = diff
                    best_exp = exp_str
                    best_dte = dte

        if best_exp is None:
            return None

        # Get option chain
        chain = t.option_chain(best_exp)
        puts = chain.puts
        calls = chain.calls

        if puts.empty or calls.empty:
            return None

        T = best_dte / 365.0

        # Use implied volatility from ATM options as sigma estimate
        # yfinance IV is unreliable outside market hours (often 0.125 or garbage)
        atm_puts = puts.iloc[(puts["strike"] - S).abs().argsort()[:3]]
        atm_calls = calls.iloc[(calls["strike"] - S).abs().argsort()[:3]]
        ivs = pd.concat([atm_puts["impliedVolatility"], atm_calls["impliedVolatility"]])
        # Filter out placeholder IVs (yfinance returns 0.125, 0.0625 as defaults)
        valid_ivs = ivs[(ivs > 0.08) & (ivs < 3.0) & (~ivs.isin([0.125, 0.0625, 0.125009, 0.062509]))]
        if len(valid_ivs) >= 2:
            sigma = valid_ivs.median()
        else:
            # Fallback: use VIX as individual stock vol estimate
            # Individual stock vol is typically 1.2-1.5x VIX
            sigma = vix / 100.0 * 1.3
            log.debug(f"  {ticker}: using VIX-based sigma={sigma:.3f} (yfinance IV unreliable)")

        # Sanity check: sigma should be between 10% and 200%
        sigma = max(0.10, min(sigma, 2.0))

        # Find short put strike (~0.20 delta)
        put_strikes = puts["strike"].values
        short_put_K = find_strike_by_delta(S, put_strikes, T, r, sigma, PUT_DELTA, "put")
        if short_put_K is None:
            return None

        # Find short call strike (~0.20 delta)
        call_strikes = calls["strike"].values
        short_call_K = find_strike_by_delta(S, call_strikes, T, r, sigma, CALL_DELTA, "call")
        if short_call_K is None:
            return None

        # Long legs: 5% of strike further OTM
        long_put_K_target = short_put_K * (1 - WING_OFFSET_PCT)
        long_call_K_target = short_call_K * (1 + WING_OFFSET_PCT)

        # Find closest available strikes
        long_put_K = put_strikes[np.argmin(np.abs(put_strikes - long_put_K_target))]
        long_call_K = call_strikes[np.argmin(np.abs(call_strikes - long_call_K_target))]

        # Ensure proper ordering and OTM positioning
        if long_put_K >= short_put_K or long_call_K <= short_call_K:
            return None

        # Validate strikes are actually OTM (not ATM or ITM)
        # Short put should be at least 3% below spot
        if short_put_K >= S * 0.97:
            log.debug(f"  {ticker}: short put {short_put_K} too close to spot {S:.2f}")
            return None
        # Short call should be at least 3% above spot
        if short_call_K <= S * 1.03:
            log.debug(f"  {ticker}: short call {short_call_K} too close to spot {S:.2f}")
            return None

        # Get bid/ask for each leg
        def get_option_price(df, strike):
            row = df[df["strike"] == strike]
            if row.empty:
                return None, None, None
            row = row.iloc[0]
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            last = float(row.get("lastPrice", 0))
            iv = float(row.get("impliedVolatility", sigma))
            # Use midpoint if bid/ask available, else lastPrice
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                return mid, (ask - bid) / mid if mid > 0 else 1.0, iv
            elif last > 0:
                return last, 0.15, iv  # Assume 15% BA if no live quotes
            return None, None, None

        sp_price, sp_ba, sp_iv = get_option_price(puts, short_put_K)
        lp_price, lp_ba, lp_iv = get_option_price(puts, long_put_K)
        sc_price, sc_ba, sc_iv = get_option_price(calls, short_call_K)
        lc_price, lc_ba, lc_iv = get_option_price(calls, long_call_K)

        if any(p is None for p in [sp_price, lp_price, sc_price, lc_price]):
            return None

        # Net credit (midpoint)
        net_credit = (sp_price - lp_price) + (sc_price - lc_price)

        if net_credit <= 0.05:
            return None  # Not enough credit

        # Estimated BA cost (half-spread on each leg at open)
        ba_cost_per_share = (sp_price * sp_ba/2 + lp_price * lp_ba/2 +
                             sc_price * sc_ba/2 + lc_price * lc_ba/2)
        net_credit_after_ba = net_credit - ba_cost_per_share

        if net_credit_after_ba <= 0:
            return None  # BA eats all the credit

        # Margin = max of put spread width or call spread width
        put_spread_width = short_put_K - long_put_K
        call_spread_width = long_call_K - short_call_K
        margin = max(put_spread_width, call_spread_width)

        # Max loss = margin - net credit (per share)
        max_loss = margin - net_credit_after_ba

        actual_put_delta = bs_delta(S, short_put_K, T, r, sigma, "put")
        actual_call_delta = bs_delta(S, short_call_K, T, r, sigma, "call")

        return {
            "ticker": ticker,
            "expiration": best_exp,
            "dte": best_dte,
            "underlying_price": S,
            "short_put": short_put_K,
            "long_put": long_put_K,
            "short_call": short_call_K,
            "long_call": long_call_K,
            "sp_price": sp_price,
            "lp_price": lp_price,
            "sc_price": sc_price,
            "lc_price": lc_price,
            "net_credit": round(net_credit, 4),
            "ba_cost": round(ba_cost_per_share, 4),
            "net_credit_after_ba": round(net_credit_after_ba, 4),
            "margin_per_share": margin,
            "max_loss_per_share": round(max_loss, 4),
            "put_delta": round(actual_put_delta, 3),
            "call_delta": round(actual_call_delta, 3),
            "avg_ba_pct": round(np.mean([sp_ba, lp_ba, sc_ba, lc_ba]) * 100, 1),
            "sigma": round(sigma, 4),
        }

    except Exception as e:
        log.debug(f"  {ticker}: scan failed: {e}")
        return None


def mark_position_to_market(pos: dict, r: float) -> dict:
    """
    Mark an existing position to market using BS model.
    Returns updated position with current P&L.
    """
    try:
        t = yf.Ticker(pos["ticker"])
        hist = t.history(period="1d")
        if hist.empty:
            return pos
        S = float(hist["Close"].iloc[-1])

        exp_date = datetime.strptime(pos["expiration"], "%Y-%m-%d").date()
        dte = (exp_date - date.today()).days
        T = max(dte / 365.0, 1/365.0)
        sigma = pos.get("sigma", 0.30)

        # Current theoretical prices
        sp_now = bs_price(S, pos["short_put"], T, r, sigma, "put")
        lp_now = bs_price(S, pos["long_put"], T, r, sigma, "put")
        sc_now = bs_price(S, pos["short_call"], T, r, sigma, "call")
        lc_now = bs_price(S, pos["long_call"], T, r, sigma, "call")

        cost_to_close = (sp_now - lp_now) + (sc_now - lc_now)
        # BA cost to close
        ba_close = sum(p * 0.06 for p in [sp_now, lp_now, sc_now, lc_now])  # Use 6% for close (tighter near expiry)

        unrealized_pnl = (pos["net_credit_after_ba"] - cost_to_close - ba_close) * 100 * pos["contracts"]

        pos["current_price"] = round(S, 2)
        pos["current_cost_to_close"] = round(cost_to_close, 4)
        pos["unrealized_pnl"] = round(unrealized_pnl, 2)
        pos["dte_remaining"] = dte
        pos["pnl_pct"] = round(unrealized_pnl / (pos["margin_dollar"] or 1) * 100, 1)

        # Check profit take / stop loss
        credit_dollar = pos["net_credit_after_ba"] * 100 * pos["contracts"]
        max_loss_dollar = pos["max_loss_per_share"] * 100 * pos["contracts"]

        if unrealized_pnl >= credit_dollar * PROFIT_TAKE:
            pos["action"] = "CLOSE_PROFIT"
        elif unrealized_pnl <= -credit_dollar * STOP_LOSS_MULT:
            pos["action"] = "CLOSE_STOP"
        elif dte <= 0:
            pos["action"] = "EXPIRED"
        else:
            pos["action"] = "HOLD"

    except Exception as e:
        log.warning(f"  MTM failed for {pos['ticker']}: {e}")
        pos["action"] = "HOLD"

    return pos


def close_position(state: dict, pid: str, reason: str) -> float:
    """Close a position and return realized P&L."""
    pos = state["positions"][pid]
    pnl = pos.get("unrealized_pnl", 0)

    # Record trade
    trade = {
        "date": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
        "ticker": pos["ticker"],
        "action": "CLOSE",
        "reason": reason,
        "expiration": pos["expiration"],
        "short_put": pos["short_put"],
        "long_put": pos["long_put"],
        "short_call": pos["short_call"],
        "long_call": pos["long_call"],
        "contracts": pos["contracts"],
        "net_credit": pos["net_credit"],
        "pnl": round(pnl, 2),
        "pnl_pct": pos.get("pnl_pct", 0),
        "hold_days": (date.today() - datetime.strptime(pos["open_date"], "%Y-%m-%d").date()).days,
    }
    append_history(trade)

    state["cash"] += pnl + pos["margin_dollar"]  # Return margin + P&L
    state["total_trades"] += 1
    state["total_pnl"] += pnl
    if pnl > 0:
        state["total_wins"] += 1

    log.info(f"  CLOSED {pos['ticker']} IC ({reason}): P&L ${pnl:+.2f} "
             f"({pos.get('pnl_pct', 0):+.1f}%)")

    del state["positions"][pid]
    return pnl


def open_position(state: dict, trade: dict, vix_mult: float):
    """Open a new iron condor position."""
    portfolio_value = state["cash"] + sum(
        p.get("margin_dollar", 0) for p in state["positions"].values()
    )

    # Position sizing: 4% of portfolio, adjusted by VIX
    alloc = portfolio_value * PER_NAME_PCT * vix_mult
    margin_per_contract = trade["margin_per_share"] * 100
    if margin_per_contract <= 0:
        return

    contracts = max(1, int(alloc / margin_per_contract))
    margin_dollar = contracts * margin_per_contract
    credit_dollar = trade["net_credit_after_ba"] * 100 * contracts

    if margin_dollar > state["cash"]:
        contracts = max(1, int(state["cash"] / margin_per_contract))
        margin_dollar = contracts * margin_per_contract
        if margin_dollar > state["cash"]:
            log.info(f"  SKIP {trade['ticker']}: insufficient cash (${state['cash']:.0f})")
            return

    pid = str(state["next_id"])
    state["next_id"] += 1

    pos = {
        "ticker": trade["ticker"],
        "open_date": date.today().strftime("%Y-%m-%d"),
        "expiration": trade["expiration"],
        "dte_at_open": trade["dte"],
        "underlying_at_open": trade["underlying_price"],
        "short_put": trade["short_put"],
        "long_put": trade["long_put"],
        "short_call": trade["short_call"],
        "long_call": trade["long_call"],
        "contracts": contracts,
        "net_credit": trade["net_credit"],
        "net_credit_after_ba": trade["net_credit_after_ba"],
        "margin_per_share": trade["margin_per_share"],
        "margin_dollar": margin_dollar,
        "credit_dollar": credit_dollar,
        "max_loss_per_share": trade["max_loss_per_share"],
        "sigma": trade["sigma"],
        "put_delta": trade["put_delta"],
        "call_delta": trade["call_delta"],
        "avg_ba_pct": trade["avg_ba_pct"],
    }

    state["positions"][pid] = pos
    state["cash"] -= margin_dollar

    # Record trade
    trade_record = {
        "date": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
        "ticker": trade["ticker"],
        "action": "OPEN",
        "reason": f"IC delta={trade['put_delta']:.2f}/{trade['call_delta']:.2f}",
        "expiration": trade["expiration"],
        "short_put": trade["short_put"],
        "long_put": trade["long_put"],
        "short_call": trade["short_call"],
        "long_call": trade["long_call"],
        "contracts": contracts,
        "net_credit": trade["net_credit"],
        "pnl": 0,
        "pnl_pct": 0,
        "hold_days": 0,
    }
    append_history(trade_record)

    log.info(f"  OPENED {trade['ticker']} IC: {trade['short_put']}/{trade['long_put']}P "
             f"{trade['short_call']}/{trade['long_call']}C x{contracts} "
             f"credit=${credit_dollar:.0f} margin=${margin_dollar:.0f} "
             f"BA={trade['avg_ba_pct']:.1f}%")


# ── Main loop ─────────────────────────────────────────────────────────────

def run_daily():
    """Execute one daily cycle of the iron condor paper engine."""
    now = datetime.now(ET)
    log.info(f"\n{'='*60}")
    log.info(f"Iron Condor Paper Engine — {now.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"{'='*60}")

    state = load_state()
    vix = get_vix()
    r = get_risk_free_rate()
    vix_mult = vix_sizing_mult(vix)

    portfolio_value = state["cash"] + sum(
        p.get("margin_dollar", 0) for p in state["positions"].values()
    )
    n_open = len(state["positions"])

    log.info(f"Portfolio: ${portfolio_value:,.0f} | Cash: ${state['cash']:,.0f} | "
             f"Open: {n_open}/{MAX_CONCURRENT} | VIX: {vix:.1f} (mult={vix_mult:.0%})")

    # ── Step 1: Mark existing positions to market ──
    closes = []
    if state["positions"]:
        log.info("\n--- Marking positions to market ---")
        for pid in list(state["positions"].keys()):
            pos = mark_position_to_market(state["positions"][pid], r)
            state["positions"][pid] = pos

            status = f"  #{pid} {pos['ticker']}: P&L ${pos.get('unrealized_pnl', 0):+.2f} "
            status += f"({pos.get('pnl_pct', 0):+.1f}%) DTE={pos.get('dte_remaining', '?')}"

            if pos.get("action") in ("CLOSE_PROFIT", "CLOSE_STOP", "EXPIRED"):
                status += f" → {pos['action']}"
                closes.append((pid, pos["action"]))

            log.info(status)

    # ── Step 2: Close positions that hit targets ──
    for pid, reason in closes:
        close_position(state, pid, reason)

    # ── Step 3: Scan for new opportunities ──
    n_open = len(state["positions"])
    slots_available = MAX_CONCURRENT - n_open

    if slots_available <= 0:
        log.info(f"\nNo slots available ({n_open}/{MAX_CONCURRENT})")
    elif vix_mult == 0:
        log.info(f"\nVIX gate CLOSED (VIX={vix:.1f} > 30)")
    else:
        log.info(f"\n--- Scanning for new ICs ({slots_available} slots) ---")

        # Get current prices for universe
        tickers_to_scan = [t for t in UNIVERSE
                           if t not in {p["ticker"] for p in state["positions"].values()}]

        candidates = []
        for ticker in tickers_to_scan:
            try:
                hist = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if hist.empty:
                    continue
                S = float(hist["Close"].iloc[-1])

                # Skip if earnings soon
                if has_earnings_soon(ticker):
                    log.info(f"  {ticker}: SKIP (earnings within {EARNINGS_BUFFER}d)")
                    continue

                ic = scan_for_ic(ticker, S, vix, r)
                if ic:
                    candidates.append(ic)
                    log.info(f"  {ticker}: IC found — credit ${ic['net_credit_after_ba']:.2f}/sh "
                             f"BA={ic['avg_ba_pct']:.1f}% delta={ic['put_delta']:.2f}/{ic['call_delta']:.2f}")
                else:
                    log.debug(f"  {ticker}: no suitable IC")

            except Exception as e:
                log.debug(f"  {ticker}: error: {e}")

        # Rank by credit/margin ratio (best risk/reward first)
        candidates.sort(key=lambda x: x["net_credit_after_ba"] / max(x["margin_per_share"], 0.01),
                        reverse=True)

        opened = 0
        for ic in candidates:
            if opened >= slots_available:
                break
            open_position(state, ic, vix_mult)
            opened += 1

        if opened == 0 and candidates:
            log.info("  No positions opened (insufficient cash or margin)")
        elif not candidates:
            log.info("  No suitable IC opportunities found")

    # ── Step 4: Summary ──
    portfolio_value = state["cash"] + sum(
        p.get("margin_dollar", 0) for p in state["positions"].values()
    )
    total_unrealized = sum(
        p.get("unrealized_pnl", 0) for p in state["positions"].values()
    )
    n_open = len(state["positions"])
    total_return = (portfolio_value + total_unrealized - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr = state["total_wins"] / max(state["total_trades"], 1) * 100

    log.info(f"\n--- Summary ---")
    log.info(f"Portfolio: ${portfolio_value:,.0f} (return: {total_return:+.1f}%)")
    log.info(f"Unrealized: ${total_unrealized:+,.0f}")
    log.info(f"Open positions: {n_open}/{MAX_CONCURRENT}")
    log.info(f"Total trades: {state['total_trades']} | Win rate: {wr:.0f}% | "
             f"Total realized P&L: ${state['total_pnl']:+,.0f}")

    save_state(state)

    # Send alert if significant events
    if closes:
        close_summary = ", ".join(
            f"{state.get('_last_closed', {}).get(pid, {}).get('ticker', '?')}"
            for pid, _ in closes
        )
        # Build a clean summary
        msg = (f"**IC Paper Engine Update**\n"
               f"Portfolio: ${portfolio_value:,.0f} ({total_return:+.1f}%)\n"
               f"Closed {len(closes)} position(s) | Open: {n_open}\n"
               f"Win rate: {wr:.0f}% over {state['total_trades']} trades")
        send_alert(msg)

    return state


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    """Single execution — designed to run via PM2 cron at 15:55 ET."""
    try:
        state = run_daily()
        log.info("Run complete.")
    except Exception as e:
        log.error(f"FATAL: {traceback.format_exc()}")
        send_alert(f"IC Paper Engine ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
