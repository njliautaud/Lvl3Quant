#!/usr/bin/env python3
"""
Earnings IV Crush Paper Engine
================================
Validates the earnings IV crush strategy with REAL daily option prices.

Strategy:
  - 1-2 days before earnings: sell iron condor (sell 20-delta put + sell 20-delta call,
    buy 5-delta put + buy 5-delta call for wings)
  - Morning after earnings: buy back the iron condor
  - Edge: IV collapses after earnings are announced, option prices drop, we keep the diff

Pricing:
  - Primary: Alpaca real market quotes via alpaca_options_pricing.get_ic_quotes()
  - Fallback: Black-Scholes with realized vol (flagged as "BS-priced")

Universe: 71 Dolt tickers (prices.parquet)

Starting NAV: $100,000
Max risk per trade: 2% NAV (iron condor max loss = spread width - credit)
Max concurrent positions: 5
Commission: $0 (Robinhood, HC #694)

PM2 / cron: run daily at 9:35 AM ET on weekdays.
State: /home/jupiter/Lvl3Quant/data/paper_engines/earnings_crush/state.json
Idempotent: safe to call multiple times per day.

CLI:
    python3 -m live_trading_linux.earnings_crush_paper --run     # normal daily run
    python3 -m live_trading_linux.earnings_crush_paper --smoke   # one-shot smoke test
    python3 -m live_trading_linux.earnings_crush_paper --status  # print current state
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "data" / "paper_engines" / "earnings_crush"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
EQUITY_FILE = STATE_DIR / "equity.csv"
EARNINGS_CACHE_FILE = STATE_DIR / "earnings_cache.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [EARN-CRUSH] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "earnings_crush_paper.log")),
    ],
)
log = logging.getLogger("EARN-CRUSH")

# ── Configuration ──────────────────────────────────────────────────────────────
STARTING_NAV = 100_000.0
MAX_RISK_PCT = 0.02          # 2% of NAV max loss per trade
MAX_POSITIONS = 5            # max concurrent iron condors

# Iron condor structure
DELTA_SHORT = 0.20           # sell 20-delta put + call
DELTA_WING = 0.05            # buy 5-delta put + call (wings)
DTE_ENTRY_MIN = 1            # open 1-2 days before earnings
DTE_ENTRY_MAX = 2
DTE_EXPIRY_OFFSET = 2        # use expiry 2 days after earnings (captures post-earnings)

# IV filter: only enter if IV rank >= 40th pctile (earnings boost IV)
IV_RANK_MIN = 0.40
# Min credit to bother (per share)
MIN_NET_CREDIT = 0.20

RISK_FREE = 0.04

# ── Universe: 71 Dolt tickers (BRK-B excluded from options — no options traded) ──
DOLT_UNIVERSE = [
    'AAPL', 'ABBV', 'ABNB', 'ADBE', 'AMD', 'AMZN', 'ARM', 'AXP',
    'BA', 'BAC', 'BLK', 'C', 'CAT', 'CL', 'COIN', 'COST', 'CRM',
    'CRWD', 'CVX', 'DDOG', 'DE', 'DIS', 'F', 'GE', 'GM', 'GOOGL',
    'GS', 'HD', 'HOOD', 'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'LOW',
    'MA', 'MCD', 'META', 'MRNA', 'MS', 'MSFT', 'NFLX', 'NOW', 'NVDA',
    'ORCL', 'OXY', 'PANW', 'PEP', 'PFE', 'PG', 'PLTR', 'PYPL', 'RTX',
    'SBUX', 'SCHW', 'SHOP', 'SLB', 'SMCI', 'T', 'TGT', 'TMUS', 'TSLA',
    'UBER', 'UNH', 'V', 'VZ', 'WFC', 'WMT', 'XOM',
]

# ── Alpaca options pricing (real quotes) ───────────────────────────────────────
_alpaca_available = False
_aop = None

def _init_alpaca():
    global _alpaca_available, _aop
    try:
        try:
            from live_trading_linux import alpaca_options_pricing as aop
        except ImportError:
            import alpaca_options_pricing as aop
        _aop = aop
        _alpaca_available = True
        log.info("Alpaca options pricing loaded — real quotes enabled")
    except Exception as e:
        _alpaca_available = False
        log.warning(f"Alpaca not available, falling back to BS pricing: {e}")


# ── Black-Scholes helpers ──────────────────────────────────────────────────────
def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, kind: str = "put") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-RISK_FREE * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-RISK_FREE * T) * _Phi(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, kind: str = "put") -> float:
    if T <= 0 or sigma <= 0:
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return _Phi(d1) - 1.0
    return _Phi(d1)


def find_strike_for_delta(S: float, sigma: float, T: float, target_delta: float,
                           kind: str = "put") -> float:
    """Binary search for strike at target absolute delta."""
    if T <= 0 or sigma <= 0:
        return S
    if kind == "put":
        lo, hi = S * 0.30, S * 1.00
    else:
        lo, hi = S * 1.00, S * 2.00
    for _ in range(60):
        K = (lo + hi) / 2.0
        d = bs_delta(S, K, T, sigma, kind)
        delta_abs = abs(d)
        if delta_abs > target_delta:
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2  # nearest $0.50


def bs_ic_price(S: float, sigma: float, T: float) -> Dict[str, Any]:
    """
    Price an iron condor via Black-Scholes at given delta targets.
    Returns strikes, individual leg prices, net credit, max loss, wing width.
    """
    short_put_K = find_strike_for_delta(S, sigma, T, DELTA_SHORT, "put")
    long_put_K = find_strike_for_delta(S, sigma, T, DELTA_WING, "put")
    short_call_K = find_strike_for_delta(S, sigma, T, DELTA_SHORT, "call")
    long_call_K = find_strike_for_delta(S, sigma, T, DELTA_WING, "call")

    sp = bs_price(S, short_put_K, T, sigma, "put")
    lp = bs_price(S, long_put_K, T, sigma, "put")
    sc = bs_price(S, short_call_K, T, sigma, "call")
    lc = bs_price(S, long_call_K, T, sigma, "call")

    net_credit = (sp - lp) + (sc - lc)
    put_width = short_put_K - long_put_K
    call_width = long_call_K - short_call_K
    wing_width = min(put_width, call_width)
    max_loss = wing_width - net_credit

    return {
        "source": "BS",
        "net_credit_mid": round(net_credit, 3),
        "net_credit_natural": round(net_credit * 0.85, 3),   # approx natural fill (15% worse)
        "max_loss": round(max(max_loss, 0.0), 3),
        "wing_width": round(wing_width, 3),
        "short_put_K": short_put_K,
        "long_put_K": long_put_K,
        "short_call_K": short_call_K,
        "long_call_K": long_call_K,
        "short_put_price": round(sp, 4),
        "long_put_price": round(lp, 4),
        "short_call_price": round(sc, 4),
        "long_call_price": round(lc, 4),
        "sigma_used": round(sigma, 4),
    }


# ── Earnings calendar ──────────────────────────────────────────────────────────
def _load_static_earnings_cache() -> Dict[str, List[str]]:
    """Load previously fetched earnings dates from cache file."""
    if EARNINGS_CACHE_FILE.exists():
        try:
            with open(EARNINGS_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_earnings_cache(cache: Dict[str, List[str]]):
    with open(EARNINGS_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_earnings_calendar_yf(ticker: str) -> List[date]:
    """
    Fetch upcoming earnings dates from yfinance.
    yf.Ticker.calendar returns a dict with 'Earnings Date' key.
    yf.Ticker.earnings_dates returns a DataFrame with upcoming + recent dates.
    We try both approaches and merge.
    """
    dates = set()
    try:
        t = yf.Ticker(ticker)
        # Approach 1: calendar (next earnings)
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        if isinstance(ed, (list, tuple)):
                            for d in ed:
                                try:
                                    dates.add(pd.Timestamp(d).date())
                                except Exception:
                                    pass
                        else:
                            try:
                                dates.add(pd.Timestamp(ed).date())
                            except Exception:
                                pass
                elif hasattr(cal, "loc"):
                    # DataFrame format (older yfinance)
                    for col in ["Earnings Date", "0", "1"]:
                        try:
                            v = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else None
                            if v is not None:
                                dates.add(pd.Timestamp(v.iloc[0]).date())
                        except Exception:
                            pass
        except Exception as e:
            log.debug(f"{ticker} calendar fetch error: {e}")

        # Approach 2: earnings_dates (multiple upcoming quarters)
        try:
            edf = t.earnings_dates
            if edf is not None and not edf.empty:
                today = date.today()
                for idx in edf.index:
                    try:
                        d = pd.Timestamp(idx).date()
                        if d >= today - timedelta(days=5):  # keep recent + future
                            dates.add(d)
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"{ticker} earnings_dates fetch error: {e}")

    except Exception as e:
        log.warning(f"Error fetching earnings for {ticker}: {e}")

    return sorted(dates)


def refresh_earnings_calendar(universe: List[str],
                               force: bool = False) -> Dict[str, List[str]]:
    """
    Refresh earnings calendar for all tickers.
    Cache is refreshed once per day (or forced).
    Returns dict: ticker -> list of date strings ("YYYY-MM-DD").
    """
    cache = _load_static_earnings_cache()
    today_str = date.today().isoformat()
    last_refresh = cache.get("_meta", {}).get("refresh_date", "")

    if not force and last_refresh == today_str:
        log.info(f"Earnings calendar: using today's cache ({len(cache)-1} tickers)")
        return {k: v for k, v in cache.items() if k != "_meta"}

    log.info(f"Refreshing earnings calendar for {len(universe)} tickers...")
    new_cache: Dict[str, List[str]] = {}
    errors = 0

    for i, ticker in enumerate(universe):
        try:
            dates = fetch_earnings_calendar_yf(ticker)
            new_cache[ticker] = [d.isoformat() for d in dates]
            if dates:
                log.debug(f"  {ticker}: {[d.isoformat() for d in dates]}")
        except Exception as e:
            log.warning(f"  {ticker}: fetch failed — {e}")
            # Preserve stale cache entry
            if ticker in cache:
                new_cache[ticker] = cache[ticker]
            errors += 1

        if (i + 1) % 20 == 0:
            log.info(f"  ...{i+1}/{len(universe)} done")
        time.sleep(0.3)  # politeness to yfinance

    new_cache["_meta"] = {"refresh_date": today_str, "errors": errors}
    _save_earnings_cache(new_cache)

    with_dates = sum(1 for k, v in new_cache.items() if k != "_meta" and v)
    log.info(f"Earnings calendar refreshed: {with_dates}/{len(universe)} tickers have dates "
             f"({errors} errors)")
    return {k: v for k, v in new_cache.items() if k != "_meta"}


def get_earnings_window(earnings_calendar: Dict[str, List[str]],
                         check_date: Optional[date] = None
                         ) -> Dict[str, date]:
    """
    Return tickers with earnings in the next DTE_ENTRY_MIN to DTE_ENTRY_MAX days.
    Returns dict: ticker -> earnings_date.
    """
    if check_date is None:
        check_date = date.today()

    window: Dict[str, date] = {}
    for ticker, date_strs in earnings_calendar.items():
        for ds in date_strs:
            try:
                ed = date.fromisoformat(ds)
                days_until = (ed - check_date).days
                if DTE_ENTRY_MIN <= days_until <= DTE_ENTRY_MAX:
                    window[ticker] = ed
                    break
            except ValueError:
                pass
    return window


def get_post_earnings_closers(earnings_calendar: Dict[str, List[str]],
                               open_positions: List[dict],
                               check_date: Optional[date] = None
                               ) -> List[str]:
    """
    Return tickers with positions that should be closed today
    (earnings was yesterday or earlier, and position is open).
    """
    if check_date is None:
        check_date = date.today()

    open_tickers = {p["ticker"] for p in open_positions}
    closers = []

    for pos in open_positions:
        ticker = pos["ticker"]
        earnings_date_str = pos.get("earnings_date", "")
        if not earnings_date_str:
            continue
        try:
            ed = date.fromisoformat(earnings_date_str)
        except ValueError:
            continue
        # Close morning after earnings (ed + 1 <= check_date <= ed + 3)
        days_after = (check_date - ed).days
        if 1 <= days_after <= 3:
            closers.append(ticker)

    return closers


# ── Price data ─────────────────────────────────────────────────────────────────
def get_spot_and_vol(ticker: str) -> Tuple[Optional[float], float]:
    """
    Get current price and 20-day realized vol for a ticker.
    Returns (spot, sigma). sigma defaults to 0.30 if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="30d", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None, 0.30
        spot = float(hist["Close"].iloc[-1])
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(log_ret) >= 10:
            sigma = float(log_ret.tail(20).std() * np.sqrt(252))
        else:
            sigma = 0.30
        sigma = max(0.05, min(sigma, 3.0))
        return spot, sigma
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
        return None, 0.30


def get_iv_rank(ticker: str, current_sigma: float) -> float:
    """
    Estimate IV rank using rolling historical realized vol as proxy for IV history.
    Returns percentile rank [0, 1].
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="260d", auto_adjust=True)
        if len(hist) < 40:
            return 0.5
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
        vol_hist = rolling_vol.dropna().values
        if len(vol_hist) < 20:
            return 0.5
        rank = float(np.sum(vol_hist <= current_sigma) / len(vol_hist))
        return rank
    except Exception as e:
        log.debug(f"IV rank failed for {ticker}: {e}")
        return 0.5


# ── IC pricing (Alpaca real or BS fallback) ────────────────────────────────────
def find_expiry_date(entry_date: date, earnings_date: date) -> date:
    """
    Find the best expiry for the iron condor.
    Target: first Friday on or after (earnings_date + DTE_EXPIRY_OFFSET).
    This ensures the option expires after earnings volatility settles.
    """
    target_min = earnings_date + timedelta(days=DTE_EXPIRY_OFFSET)
    # Find next Friday >= target_min
    d = target_min
    while d.weekday() != 4:  # 4 = Friday
        d += timedelta(days=1)
    # Check it's at least DTE_ENTRY_MAX days from entry
    if (d - entry_date).days < 2:
        d += timedelta(days=7)
    return d


def price_iron_condor(ticker: str, spot: float, sigma: float,
                       expiry: date, entry_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Price an iron condor.
    1. Try Alpaca real market quotes (during market hours).
    2. Fall back to BS with sigma as IV.

    Returns full pricing dict with 'source' key ('alpaca' or 'BS').
    """
    if entry_date is None:
        entry_date = date.today()

    T = max((expiry - entry_date).days, 1) / 365.0

    # Determine wing width from BS (use 5-delta strikes)
    short_put_K = find_strike_for_delta(spot, sigma, T, DELTA_SHORT, "put")
    long_put_K = find_strike_for_delta(spot, sigma, T, DELTA_WING, "put")
    short_call_K = find_strike_for_delta(spot, sigma, T, DELTA_SHORT, "call")
    long_call_K = find_strike_for_delta(spot, sigma, T, DELTA_WING, "call")
    put_width = short_put_K - long_put_K
    call_width = long_call_K - short_call_K
    wing_width = min(put_width, call_width)

    # Try Alpaca first
    if _alpaca_available and _aop is not None:
        try:
            result = _aop.get_ic_quotes(
                ticker=ticker,
                put_strike=short_put_K,
                call_strike=short_call_K,
                wing_width=wing_width,
                expiry=expiry,
            )
            if result and result.get("net_credit_mid", 0) > 0:
                result["source"] = "alpaca"
                result["short_put_K"] = short_put_K
                result["long_put_K"] = long_put_K
                result["short_call_K"] = short_call_K
                result["long_call_K"] = long_call_K
                result["wing_width"] = round(wing_width, 3)
                result["sigma_used"] = round(sigma, 4)
                log.info(f"  {ticker}: Alpaca IC quote — credit=${result['net_credit_mid']:.3f}, "
                         f"max_loss=${result['max_loss']:.3f}")
                return result
            else:
                log.info(f"  {ticker}: Alpaca returned zero credit — falling back to BS")
        except Exception as e:
            log.warning(f"  {ticker}: Alpaca IC pricing failed ({e}) — falling back to BS")

    # BS fallback
    bs_result = bs_ic_price(spot, sigma, T)
    log.info(f"  {ticker}: BS-priced IC — credit=${bs_result['net_credit_mid']:.3f} "
             f"[sigma={sigma:.2%}] (NOTE: BS estimate, not real quote)")
    return bs_result


# ── State management ───────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"State load error: {e} — using fresh state")

    return {
        "nav": STARTING_NAV,
        "cash": STARTING_NAV,
        "positions": [],        # list of open IC positions
        "closed_trades": [],    # history of closed trades
        "daily_equity": [],     # equity curve
        "trade_count": 0,
        "start_date": date.today().isoformat(),
        "last_run_date": None,
        "version": "v1",
    }


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


def compute_nav(state: dict, current_prices: Dict[str, float]) -> float:
    """Mark-to-market NAV. For open ICs, use BS MTM on current spot."""
    nav = state["cash"]
    today = date.today()

    for pos in state["positions"]:
        ticker = pos["ticker"]
        spot = current_prices.get(ticker)
        if spot is None:
            # No price — use entry credit as proxy (conservative)
            nav += pos.get("net_credit", 0) * 100 * pos.get("contracts", 1)
            continue

        expiry = date.fromisoformat(pos["expiry"])
        sigma = pos.get("sigma_entry", 0.25)
        T = max((expiry - today).days, 0) / 365.0

        # Current BS value of the IC (what it would cost to buy back)
        short_put_val = bs_price(spot, pos["short_put_K"], T, sigma, "put")
        long_put_val = bs_price(spot, pos["long_put_K"], T, sigma, "put")
        short_call_val = bs_price(spot, pos["short_call_K"], T, sigma, "call")
        long_call_val = bs_price(spot, pos["long_call_K"], T, sigma, "call")
        current_ic_cost = (short_put_val - long_put_val) + (short_call_val - long_call_val)

        # P&L per share = entry credit - current buyback cost
        unrealized_per_sh = pos["net_credit"] - current_ic_cost
        if unrealized_per_sh != unrealized_per_sh:  # NaN check
            # BS returned NaN (e.g. pre-market, no prices) — fall back to credit
            nav += pos.get("net_credit", 0) * 100 * pos.get("contracts", 1)
        else:
            nav += unrealized_per_sh * 100 * pos.get("contracts", 1)

    return nav


def log_trade(trade: dict):
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def log_equity(state: dict, nav: float):
    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, "w") as f:
            f.write("date,nav,cash,open_positions,realized_pnl\n")
    with open(EQUITY_FILE, "a") as f:
        realized = sum(t["pnl"] for t in state.get("closed_trades", []))
        f.write(f"{date.today().isoformat()},{nav:.2f},{state['cash']:.2f},"
                f"{len(state['positions'])},{realized:.2f}\n")


# ── Core logic: open positions ─────────────────────────────────────────────────
def open_positions(state: dict, earnings_window: Dict[str, date]):
    """
    For each ticker with upcoming earnings, attempt to open an iron condor.
    """
    today = date.today()
    open_tickers = {p["ticker"] for p in state["positions"]}
    n_open = len(state["positions"])

    for ticker, earnings_date in earnings_window.items():
        if ticker in open_tickers:
            log.info(f"  {ticker}: already have open position — skip")
            continue

        if n_open >= MAX_POSITIONS:
            log.info(f"Max positions ({MAX_POSITIONS}) reached — no more entries today")
            break

        log.info(f"Evaluating {ticker} (earnings {earnings_date})")

        # Get spot and vol
        spot, sigma = get_spot_and_vol(ticker)
        if spot is None:
            log.warning(f"  {ticker}: no spot price — skip")
            continue

        if spot < 5.0:
            log.info(f"  {ticker}: spot ${spot:.2f} too low — skip")
            continue

        # IV rank filter (only trade if vol is elevated)
        iv_rank = get_iv_rank(ticker, sigma)
        if iv_rank < IV_RANK_MIN:
            log.info(f"  {ticker}: IV rank {iv_rank:.2f} < {IV_RANK_MIN} — skipping low-IV trade")
            continue

        # Find expiry
        expiry = find_expiry_date(today, earnings_date)
        log.info(f"  {ticker}: spot=${spot:.2f}, sigma={sigma:.2%}, "
                 f"iv_rank={iv_rank:.2f}, expiry={expiry}")

        # Price the iron condor
        ic = price_iron_condor(ticker, spot, sigma, expiry, today)

        net_credit = ic.get("net_credit_mid", 0.0)
        max_loss_per_sh = ic.get("max_loss", ic.get("wing_width", 5.0) - net_credit)
        wing_width = ic.get("wing_width", 5.0)

        if net_credit < MIN_NET_CREDIT:
            log.info(f"  {ticker}: net credit ${net_credit:.3f} < min ${MIN_NET_CREDIT:.3f} — skip")
            continue

        if max_loss_per_sh <= 0:
            log.warning(f"  {ticker}: max_loss <= 0 (pricing error) — skip")
            continue

        # Position sizing: max 2% of NAV risk
        nav = compute_nav(state, {})
        max_dollar_risk = nav * MAX_RISK_PCT
        # max_loss_per_sh is per share, per contract = 100 shares
        contracts = max(1, int(max_dollar_risk / (max_loss_per_sh * 100)))

        # Check we have enough cash (hold margin = wing_width * 100 * contracts)
        margin_required = wing_width * 100 * contracts
        if margin_required > state["cash"] * 0.90:
            contracts = max(1, int(state["cash"] * 0.90 / (wing_width * 100)))
            margin_required = wing_width * 100 * contracts

        if contracts < 1 or margin_required > state["cash"]:
            log.info(f"  {ticker}: insufficient cash (need ${margin_required:.0f}, "
                     f"have ${state['cash']:.0f}) — skip")
            continue

        # Credit received
        total_credit = net_credit * 100 * contracts

        # Open position
        pos = {
            "ticker": ticker,
            "earnings_date": earnings_date.isoformat(),
            "expiry": expiry.isoformat(),
            "entry_date": today.isoformat(),
            "short_put_K": ic["short_put_K"],
            "long_put_K": ic["long_put_K"],
            "short_call_K": ic["short_call_K"],
            "long_call_K": ic["long_call_K"],
            "wing_width": wing_width,
            "net_credit": round(net_credit, 4),
            "max_loss_per_sh": round(max_loss_per_sh, 4),
            "contracts": contracts,
            "margin_held": round(margin_required, 2),
            "total_credit": round(total_credit, 2),
            "spot_entry": round(spot, 2),
            "sigma_entry": round(sigma, 4),
            "iv_rank_entry": round(iv_rank, 3),
            "pricing_source": ic.get("source", "BS"),
        }

        state["positions"].append(pos)
        state["cash"] -= margin_required
        state["cash"] += total_credit

        n_open += 1
        open_tickers.add(ticker)

        log.info(
            f"  OPENED IC {ticker} | earnings={earnings_date} | expiry={expiry} | "
            f"strikes: {pos['long_put_K']:.0f}/{pos['short_put_K']:.0f}/"
            f"{pos['short_call_K']:.0f}/{pos['long_call_K']:.0f} | "
            f"credit=${net_credit:.3f}/sh | contracts={contracts} | "
            f"max_loss=${max_loss_per_sh*100*contracts:.0f} | "
            f"source={ic.get('source','BS')}"
        )

        trade_rec = {
            "action": "OPEN_IC",
            "ts": datetime.now(timezone.utc).isoformat(),
            **pos,
        }
        log_trade(trade_rec)

        state["trade_count"] += 1


# ── Core logic: close positions ────────────────────────────────────────────────
def close_positions(state: dict, tickers_to_close: List[str]):
    """
    Buy back iron condors for tickers in tickers_to_close.
    Uses Alpaca real quotes for buyback if available, else BS.
    """
    today = date.today()
    remaining = []

    for pos in state["positions"]:
        ticker = pos["ticker"]

        if ticker not in tickers_to_close:
            # Check if expired
            expiry = date.fromisoformat(pos["expiry"])
            if today > expiry:
                log.info(f"  {ticker}: IC expired {expiry} — closing at intrinsic")
                # Expired: close at intrinsic (simplified — mark as expired)
                _close_at_expiry(state, pos)
            else:
                remaining.append(pos)
            continue

        # Get current spot and vol for buyback pricing
        log.info(f"Closing IC for {ticker} (post-earnings)")
        spot, sigma = get_spot_and_vol(ticker)
        if spot is None:
            log.warning(f"  {ticker}: no spot price for close — carrying position")
            remaining.append(pos)
            continue

        expiry = date.fromisoformat(pos["expiry"])

        # Price the buyback
        buyback = price_iron_condor(ticker, spot, sigma, expiry, today)
        buyback_cost_per_sh = buyback.get("net_credit_mid", None)

        if buyback_cost_per_sh is None or buyback_cost_per_sh < 0:
            log.warning(f"  {ticker}: invalid buyback price — using BS estimate")
            T = max((expiry - today).days, 1) / 365.0
            # Use entry sigma * 0.6 as post-earnings vol estimate (typical crush)
            post_sigma = pos["sigma_entry"] * 0.6
            bs_ic = bs_ic_price(spot, post_sigma, T)
            buyback_cost_per_sh = bs_ic["net_credit_mid"]

        # HC #722 FIX: Enforce intrinsic floor on buyback cost.
        # Alpaca/BS theoretical pricing can return < intrinsic for deep ITM
        # options, which is impossible in real markets (arbitrage).
        put_intrinsic = max(pos["short_put_K"] - spot, 0) - max(pos["long_put_K"] - spot, 0)
        call_intrinsic = max(spot - pos["short_call_K"], 0) - max(spot - pos["long_call_K"], 0)
        ic_intrinsic = put_intrinsic + call_intrinsic
        if buyback_cost_per_sh < ic_intrinsic:
            log.warning(
                f"  {ticker}: Alpaca/BS price ${buyback_cost_per_sh:.2f} < intrinsic "
                f"${ic_intrinsic:.2f} — using intrinsic floor"
            )
            buyback_cost_per_sh = ic_intrinsic

        contracts = pos["contracts"]
        total_buyback = buyback_cost_per_sh * 100 * contracts

        # P&L = credit received at entry - cost to buy back
        pnl = pos["total_credit"] - total_buyback

        # Release margin
        state["cash"] += pos["margin_held"]
        state["cash"] -= total_buyback  # pay for buyback

        iv_crush_pct = None
        sigma_exit = sigma
        if pos["sigma_entry"] > 0:
            iv_crush_pct = round((pos["sigma_entry"] - sigma_exit) / pos["sigma_entry"] * 100, 1)

        log.info(
            f"  CLOSED IC {ticker} | entry_credit=${pos['net_credit']:.3f}/sh | "
            f"buyback=${buyback_cost_per_sh:.3f}/sh | "
            f"pnl=${pnl:.2f} | iv_crush={iv_crush_pct}% | "
            f"source={buyback.get('source','BS')}"
        )

        trade_rec = {
            "action": "CLOSE_IC",
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "expiry": pos["expiry"],
            "earnings_date": pos["earnings_date"],
            "entry_date": pos["entry_date"],
            "exit_date": today.isoformat(),
            "entry_credit_per_sh": pos["net_credit"],
            "buyback_cost_per_sh": round(buyback_cost_per_sh, 4),
            "contracts": contracts,
            "pnl": round(pnl, 2),
            "iv_entry": pos["sigma_entry"],
            "iv_exit": round(sigma_exit, 4),
            "iv_crush_pct": iv_crush_pct,
            "spot_entry": pos["spot_entry"],
            "spot_exit": round(spot, 2),
            "spot_move_pct": round((spot - pos["spot_entry"]) / pos["spot_entry"] * 100, 2),
            "entry_source": pos["pricing_source"],
            "exit_source": buyback.get("source", "BS"),
        }
        log_trade(trade_rec)

        state["closed_trades"].append({
            "ticker": ticker,
            "earnings_date": pos["earnings_date"],
            "entry_date": pos["entry_date"],
            "exit_date": today.isoformat(),
            "pnl": round(pnl, 2),
            "iv_crush_pct": iv_crush_pct,
        })
        state["trade_count"] += 1

    state["positions"] = remaining


def _close_at_expiry(state: dict, pos: dict):
    """Close an expired iron condor at intrinsic value."""
    today = date.today()
    spot, _ = get_spot_and_vol(pos["ticker"])
    if spot is None:
        spot = pos["spot_entry"]

    # Intrinsic value: the IC loses if stock moved outside the short strikes
    short_put_K = pos["short_put_K"]
    short_call_K = pos["short_call_K"]
    long_put_K = pos["long_put_K"]
    long_call_K = pos["long_call_K"]

    put_intrinsic = max(short_put_K - spot, 0) - max(long_put_K - spot, 0)
    call_intrinsic = max(spot - short_call_K, 0) - max(spot - long_call_K, 0)
    buyback_intrinsic_per_sh = put_intrinsic + call_intrinsic

    contracts = pos["contracts"]
    total_buyback = buyback_intrinsic_per_sh * 100 * contracts
    pnl = pos["total_credit"] - total_buyback

    state["cash"] += pos["margin_held"]
    state["cash"] -= total_buyback

    log.info(f"  EXPIRED IC {pos['ticker']} | spot=${spot:.2f} | pnl=${pnl:.2f}")
    log_trade({
        "action": "EXPIRE_IC",
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": pos["ticker"],
        "pnl": round(pnl, 2),
        "spot_expiry": round(spot, 2),
        "exit_date": today.isoformat(),
    })
    state["closed_trades"].append({
        "ticker": pos["ticker"],
        "earnings_date": pos["earnings_date"],
        "entry_date": pos["entry_date"],
        "exit_date": today.isoformat(),
        "pnl": round(pnl, 2),
        "iv_crush_pct": None,
    })
    state["trade_count"] += 1


# ── Daily run ──────────────────────────────────────────────────────────────────
def run_daily():
    """Main daily cycle: close post-earnings + open pre-earnings ICs."""
    today = date.today()
    log.info("=" * 70)
    log.info(f"EARNINGS IV CRUSH — DAILY RUN — {today}")
    log.info("=" * 70)

    # Weekend check
    if today.weekday() >= 5:
        log.info("Weekend — no action")
        return

    state = load_state()

    # Idempotency: don't run twice in the same day
    if state.get("last_run_date") == today.isoformat():
        log.info(f"Already ran today ({today}) — idempotency check passed, exiting")
        return

    # Init Alpaca
    _init_alpaca()

    # ── Phase 1: Refresh earnings calendar ─────────────────────────────────────
    log.info("Phase 1: Refreshing earnings calendar...")
    earnings_cal = refresh_earnings_calendar(DOLT_UNIVERSE)

    # ── Phase 2: Close post-earnings positions ──────────────────────────────────
    log.info(f"Phase 2: Checking {len(state['positions'])} open positions for close...")
    tickers_to_close = get_post_earnings_closers(earnings_cal, state["positions"], today)

    if tickers_to_close:
        log.info(f"  Closing post-earnings: {tickers_to_close}")
        close_positions(state, tickers_to_close)
    else:
        log.info("  No positions to close today")

    # ── Phase 3: Open pre-earnings positions ────────────────────────────────────
    log.info("Phase 3: Scanning for upcoming earnings (1-2 days)...")
    earnings_window = get_earnings_window(earnings_cal, today)

    if earnings_window:
        log.info(f"  Earnings candidates: {list(earnings_window.keys())}")
        open_positions(state, earnings_window)
    else:
        log.info("  No earnings in the next 1-2 days")

    # ── Phase 4: Mark-to-market and save ────────────────────────────────────────
    # Get current prices for open positions
    current_prices = {}
    for pos in state["positions"]:
        s, _ = get_spot_and_vol(pos["ticker"])
        if s:
            current_prices[pos["ticker"]] = s

    nav = compute_nav(state, current_prices)
    state["nav"] = round(nav, 2)
    state["last_run_date"] = today.isoformat()

    realized = sum(t["pnl"] for t in state.get("closed_trades", []))
    win_trades = [t for t in state.get("closed_trades", []) if t["pnl"] > 0]
    wr = len(win_trades) / len(state["closed_trades"]) if state["closed_trades"] else 0.0

    log.info(f"\n{'='*70}")
    log.info(f"EOD SUMMARY — {today}")
    log.info(f"  NAV:        ${nav:,.2f} ({(nav/STARTING_NAV - 1)*100:+.2f}%)")
    log.info(f"  Cash:       ${state['cash']:,.2f}")
    log.info(f"  Open ICs:   {len(state['positions'])}")
    log.info(f"  Trades:     {state['trade_count']} | Win rate: {wr:.0%}")
    log.info(f"  Realized:   ${realized:,.2f}")
    if state["positions"]:
        log.info("  Open positions:")
        for p in state["positions"]:
            log.info(f"    {p['ticker']} | earnings={p['earnings_date']} | "
                     f"credit=${p['net_credit']:.3f}/sh | {p['contracts']}c | "
                     f"src={p['pricing_source']}")
    log.info("=" * 70)

    log_equity(state, nav)
    save_state(state)

    # Send to QCC / Discord via webhook
    _notify_eod(nav, realized, state)


def _notify_eod(nav: float, realized: float, state: dict):
    """Send end-of-day summary to Discord via webhook."""
    try:
        n_open = len(state["positions"])
        n_closed = len(state.get("closed_trades", []))
        win_trades = [t for t in state.get("closed_trades", []) if t["pnl"] > 0]
        wr = len(win_trades) / n_closed if n_closed else 0.0
        pnl_pct = (nav / STARTING_NAV - 1) * 100

        # Check IV crush stats from recent closed trades
        crush_trades = [t for t in state.get("closed_trades", [])
                        if t.get("iv_crush_pct") is not None]
        avg_crush = (sum(t["iv_crush_pct"] for t in crush_trades) / len(crush_trades)
                     if crush_trades else None)

        lines = [
            f"Earnings IV Crush — {date.today()}",
            f"NAV: ${nav:,.0f} ({pnl_pct:+.1f}%)",
            f"Realized P&L: ${realized:,.0f}",
            f"Trades: {n_closed} closed, {n_open} open | WR: {wr:.0%}",
        ]
        if avg_crush is not None:
            lines.append(f"Avg IV crush on close: {avg_crush:.1f}%")
        if n_open > 0:
            tickers = ", ".join(p["ticker"] for p in state["positions"])
            lines.append(f"Open: {tickers}")

        msg = "\n".join(lines)
        webhook_path = "/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
        if os.path.exists(webhook_path):
            os.system(f'node {webhook_path} "{msg.replace(chr(10), " | ")}" 2>/dev/null')
    except Exception as e:
        log.debug(f"Notification failed (non-fatal): {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def run_smoke():
    """Smoke test: check calendar + price one ticker, but don't modify state."""
    log.info("=== SMOKE TEST ===")
    _init_alpaca()

    # Test earnings calendar
    sample = DOLT_UNIVERSE[:5]
    log.info(f"Testing earnings fetch for: {sample}")
    for ticker in sample:
        dates = fetch_earnings_calendar_yf(ticker)
        log.info(f"  {ticker}: {[d.isoformat() for d in dates] or 'no upcoming dates'}")
        time.sleep(0.3)

    # Test spot price
    log.info("Testing spot + vol fetch...")
    spot, sigma = get_spot_and_vol("AAPL")
    log.info(f"  AAPL: spot=${spot}, sigma={sigma:.2%}")

    # Test IV rank
    if spot:
        iv_rank = get_iv_rank("AAPL", sigma)
        log.info(f"  AAPL IV rank: {iv_rank:.2f}")

    # Test IC pricing (don't need real expiry for smoke)
    test_expiry = date.today() + timedelta(days=7)
    if spot:
        log.info(f"Testing IC pricing for AAPL, expiry={test_expiry}...")
        ic = price_iron_condor("AAPL", spot, sigma, test_expiry)
        log.info(f"  AAPL IC: credit=${ic.get('net_credit_mid'):.3f}, "
                 f"max_loss=${ic.get('max_loss'):.3f}, source={ic.get('source')}")

    log.info("=== SMOKE TEST OK ===")


def print_status():
    """Print current state."""
    state = load_state()
    nav = state.get("nav", state.get("cash", STARTING_NAV))
    realized = sum(t["pnl"] for t in state.get("closed_trades", []))
    pnl_pct = (nav / STARTING_NAV - 1) * 100

    print(f"\nEarnings IV Crush Paper Engine — Status")
    print(f"{'='*50}")
    print(f"NAV:         ${nav:,.2f} ({pnl_pct:+.2f}%)")
    print(f"Cash:        ${state['cash']:,.2f}")
    print(f"Realized:    ${realized:,.2f}")
    print(f"Trades:      {state.get('trade_count', 0)}")
    print(f"Last run:    {state.get('last_run_date', 'never')}")
    print(f"\nOpen positions ({len(state['positions'])}):")
    for p in state["positions"]:
        print(f"  {p['ticker']:6s} | earnings={p['earnings_date']} | "
              f"expiry={p['expiry']} | credit=${p['net_credit']:.3f}/sh | "
              f"{p['contracts']}c | src={p['pricing_source']}")
    print(f"\nRecent closed trades ({min(10, len(state.get('closed_trades',[])))}):")
    for t in state.get("closed_trades", [])[-10:]:
        crush = f" | crush={t['iv_crush_pct']:.0f}%" if t.get("iv_crush_pct") else ""
        print(f"  {t['ticker']:6s} | {t.get('exit_date','')} | pnl=${t['pnl']:+.2f}{crush}")


def main():
    ap = argparse.ArgumentParser(description="Earnings IV Crush Paper Engine")
    ap.add_argument("--run", action="store_true", help="Daily run (open/close positions)")
    ap.add_argument("--smoke", action="store_true", help="Smoke test (no state changes)")
    ap.add_argument("--status", action="store_true", help="Print current state")
    ap.add_argument("--force-calendar", action="store_true",
                    help="Force-refresh earnings calendar cache")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
    elif args.status:
        print_status()
    elif args.force_calendar:
        _init_alpaca()
        cal = refresh_earnings_calendar(DOLT_UNIVERSE, force=True)
        print(f"Calendar refreshed: {len(cal)} tickers with earnings data")
        upcoming = {k: v for k, v in cal.items() if v}
        for ticker, dates in sorted(upcoming.items()):
            today = date.today()
            upcoming_dates = [d for d in dates
                              if date.fromisoformat(d) >= today]
            if upcoming_dates:
                print(f"  {ticker}: {upcoming_dates}")
    else:
        # Default: --run (PM2 / cron mode)
        run_daily()


if __name__ == "__main__":
    main()
