#!/usr/bin/env python3
"""
Bull Put Spread (BPS) Paper Engine — CONSERVATIVE Variant (HC #664)
====================================================================

OPTIMAL BPS config from d30 research (2026-07-09).
Updated from conservative defaults to research-validated optimal params.

Config (optimal from research):
  - 30-delta short leg (optimal delta from study)
  - $15-wide spreads (defined risk buffer per position)
  - 25% margin cap (optimal utilization from study)
  - Max 15 concurrent positions
  - 10-day DTE (weekly rotation, less gamma)
  - 65% profit take (optimal from study — faster premium capture)

Risk Controls:
  - Bear gate: SPY < 50d SMA -> no new spreads
  - Equity curve brake: 60d lookback, 3% DD -> scale to 25%
  - 7-day earnings buffer (yfinance, daily cache)
  - VIX hard cutoff at 30 (no new entries)
  - VIX-scaled sizing (tiered reduction)
  - Fast DD trigger: 3-day -5% trailing return halts new entries
  - Circuit breaker: 2% daily portfolio loss -> freeze new entries for 1 calendar day
  - Tier-2 mid-caps EXCLUDED (39% BS overestimate, net negative after vol-skew correction)

Cost model:
  - $0.65/contract commission PER LEG ($1.30 per spread)
  - REAL yfinance option chain bid/ask pricing (BS fallback if chain unavailable)
  - Fill quality logged to state/bps_conservative/fill_quality.jsonl

Author: Claude (2026-07-09, HC #664 — optimal config from d30 research)
"""

import json
import math
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# ── Alpaca real pricing bridge (falls back to BS when market closed) ──
try:
    try:
        from live_trading_linux.alpaca_pricing_bridge import (
            get_premium as _alpaca_get_premium, init_bridge as _init_bridge,
            log_session_summary as _log_pricing_summary,
        )
    except ImportError:
        from alpaca_pricing_bridge import (
            get_premium as _alpaca_get_premium, init_bridge as _init_bridge,
            log_session_summary as _log_pricing_summary,
        )
    _HAS_PRICING_BRIDGE = True
except Exception:
    _HAS_PRICING_BRIDGE = False

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "wheel_bps_conservative_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
FILL_QUALITY_FILE = STATE_DIR / "fill_quality.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [BPS-CONS] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_bps_conservative_paper.log")),
    ],
)
log = logging.getLogger('BPS-CONS')

# ── OPTIMAL Configuration (d30 research 2026-07-09) ──
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Spread parameters — OPTIMAL from d30 research
SPREAD_WIDTH = 15.0        # $15 wide — defined risk buffer per position
PUT_DELTA_TARGET = 0.30    # 30-delta short leg (optimal from study)
DTE_TARGET = 10            # 10-day DTE — less gamma, more time premium
DTE_MIN = 5
DTE_MAX = 14
PROFIT_TAKE_PCT = 0.65     # 65% profit take (optimal from study)
BPS_STOP_LOSS_MULT = 1.0   # Close spread when loss reaches 1x premium collected
MIN_NET_CREDIT = 0.15      # Min $0.15/share net credit

# Position sizing — HALF-KELLY from 400-trade analysis (2026-07-09)
# Full Kelly = 16.7%, Half Kelly = 8.3%. Win/loss ratio = 0.26 (small wins, large losses).
# Previous 25% cap was ~3x Kelly — too aggressive for the strategy's payoff profile.
MARGIN_CAP = 0.10          # Max 10% of NAV (slightly above half-Kelly for diversification benefit)
PER_NAME_PCT = 0.015       # Max 1.5% of NAV per name
MAX_CONCURRENT = 10        # Max 10 positions — half-Kelly sized

# Risk controls
VIX_MAX_GATE = 30.0        # Hard cutoff at VIX 30
BRAKE_LOOKBACK_DAYS = 60
BRAKE_THRESHOLD = 0.03     # 3% DD from peak triggers brake
BRAKE_SCALE = 0.25         # Scale exposure to 25% when braking

# Fast drawdown trigger (HC #662 R4 — validated by permutation test p=0.000)
# Bad days cluster 2.3x in BPS returns. When trailing 3-day return < -5%,
# halt new positions entirely. Backtest: Sharpe 4.09→4.26, MaxDD -30%→-22%.
DD_TRIGGER_LOOKBACK = 3    # 3-day trailing return window
DD_TRIGGER_THRESHOLD = -0.05  # -5% trailing return triggers halt
DD_TRIGGER_ENABLED = True  # Feature flag for A/B tracking

# VIX-scaled position sizing (HC #664 R4, VIX regime gating study 2026-07-09)
# Research finding: edge disappears above VIX 25, Sharpe goes negative above 30.
# VIX-scaled sizing: Sharpe 0.97→1.55, Sortino 0.48→0.81, MaxDD -70%→-47%
VIX_SCALE_ENABLED = True
VIX_SCALE_TIERS = [
    (15, 1.00),   # VIX < 15: full size
    (20, 0.80),   # VIX 15-20: 80%
    (25, 0.50),   # VIX 20-25: half size
    (30, 0.25),   # VIX 25-30: quarter size
    (999, 0.00),  # VIX > 30: HARD CUTOFF — no new entries
]

# Circuit breaker: 2% daily portfolio loss → freeze new entries for 1 calendar day
CIRCUIT_BREAKER_ENABLED = True
CIRCUIT_BREAKER_LOSS_PCT = 0.02   # 2% daily portfolio loss threshold
CIRCUIT_BREAKER_FREEZE_DAYS = 1   # Freeze new entries for 1 calendar day
CIRCUIT_BREAKER_FILE = STATE_DIR / "circuit_breaker.json"

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65   # Per contract per leg
SLIPPAGE_FRAC = 0.025  # Only used in BS fallback path
SLIPPAGE_MIN = 0.03    # Only used in BS fallback path

# Price filter
MIN_PRICE = 10.0
MAX_PRICE = 500.0

# ── Sigma (volatility) filter ──
# Tickers with 20d realized vol > this threshold are excluded from new entries.
# IBM (sigma=1.1), ARM, FSLR, APD all blew up with sigma > 0.9.
# Filter set at 0.8 to exclude extreme-vol names while keeping high-premium names.
SIGMA_MAX_ENTRY = 0.8   # Skip tickers with 20d annualized vol > 80%

# ── Stop-loss ticker cooldown ──
# After a stop-loss exit, block re-entry on the same ticker for 24 hours.
# Prevents IBM-style death loops (5 stop-loss cycles on July 14 = -$1,170).
STOP_LOSS_COOLDOWN_HOURS = 24

# ── Persistent loser blacklist ──
# Tickers that consistently lose across multiple engines based on live paper analysis.
# Excluded from ALL new position entries.
LOSER_BLACKLIST = {"CRSP", "AAL", "CELH"}


# ── Universe ──
def load_universe():
    """Load the full 230-ticker universe from cached data."""
    cache_dir = ROOT / "wheel_strategy_v1" / "data" / "cache"
    tickers = set()

    for pf in ["prices.parquet", "prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        path = cache_dir / pf
        if path.exists():
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                tickers |= set(df["ticker"].unique())

    tickers -= {"SPY", "^VIX", "VIX"}

    # Tier-2 removed per honest Sharpe analysis 2026-07-09 — 39% BS overestimate makes them net negative
    # (tier-1 only Sharpe 2.44 vs full universe 1.66 with tiered haircut)
    # Existing tier-2 positions are allowed to expire naturally; only new entries blocked.
    TIER2_EXCLUDED = {
        "REGN", "VRTX", "MRNA", "BIIB", "ILMN", "DXCM", "ALGN",
        "MRVL", "ON", "SWKS", "QRVO", "WOLF",
        "DVN", "FANG", "MPC", "VLO", "PSX", "HES",
        "O", "AMT", "PLD", "EQIX", "DLR",
        "EMR", "ROK", "ITW", "ETN", "IR",
    }
    tickers -= TIER2_EXCLUDED

    # Fill-quality blacklist: tickers with consistently negative real credits
    # from 688 live option chain quotes (2026-07-09 analysis).
    # These tickers have avg negative net credit or <50% viable rate at real BA spreads.
    FILL_QUALITY_BLACKLIST = {
        "IONS", "AEP", "HAL", "HON", "AMGN", "GD", "EXC", "ED",
        "CHWY", "ESS", "CL", "CARR", "EOG", "JD", "ET", "DOCU", "GOLD",
    }
    tickers -= FILL_QUALITY_BLACKLIST

    # Consistent backtest losers (from 400-trade Kelly analysis, 2026-07-09)
    # Updated 2026-07-15: added CRSP, AAL, CELH from live paper analysis
    BACKTEST_LOSERS = {"ABBV", "CROX", "CRSP", "AAL", "CELH"}
    tickers -= BACKTEST_LOSERS

    log.info(f"Universe loaded: {len(tickers)} tickers (tier-2 + fill-quality + loser blacklist excluded)")
    return sorted(tickers)


# ── Earnings Filter ──
EARNINGS_BUFFER_DAYS = 7  # Skip tickers with earnings within 7 calendar days
EARNINGS_CACHE_FILE = STATE_DIR / "earnings_cache.json"


def _fetch_yfinance_earnings(tickers):
    """Fetch next earnings dates from yfinance for a list of tickers.

    Returns dict of {ticker: "YYYY-MM-DD"} for tickers that have a known
    upcoming earnings date.  Tickers where yfinance returns nothing are
    silently omitted (caller treats missing = no earnings = OK to trade).
    """
    import yfinance as yf
    result = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # Try .calendar first (dict with 'Earnings Date')
            cal = t.calendar
            if cal is not None:
                # yfinance >= 0.2: cal is a dict
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed is not None:
                        if isinstance(ed, list) and len(ed) > 0:
                            result[ticker] = pd.Timestamp(ed[0]).strftime("%Y-%m-%d")
                        elif not isinstance(ed, list):
                            result[ticker] = pd.Timestamp(ed).strftime("%Y-%m-%d")
                        continue
                # yfinance < 0.2: cal might be a DataFrame
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    vals = cal.loc["Earnings Date"]
                    if hasattr(vals, 'iloc') and len(vals) > 0:
                        result[ticker] = pd.Timestamp(vals.iloc[0]).strftime("%Y-%m-%d")
                    else:
                        result[ticker] = pd.Timestamp(vals).strftime("%Y-%m-%d")
                    continue

            # Fallback: earnings_dates property
            edf = t.earnings_dates
            if edf is not None and not edf.empty:
                future = edf.index[edf.index >= pd.Timestamp.now() - pd.Timedelta(days=1)]
                if len(future) > 0:
                    result[ticker] = pd.Timestamp(future[0]).strftime("%Y-%m-%d")
        except Exception:
            # yfinance failure for this ticker — assume no earnings (safe default)
            pass
    return result


def load_earnings_dates_yf(universe):
    """Load earnings dates with daily yfinance cache.

    Cache strategy: store {ticker: next_earnings_date} in a JSON file.
    Refresh once per calendar day to avoid hammering yfinance.
    If yfinance fails entirely, fall back to the static parquet cache.
    """
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    cache = {}

    # Try loading existing cache
    if EARNINGS_CACHE_FILE.exists():
        try:
            with open(EARNINGS_CACHE_FILE) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # If cache is from today, reuse it
    if cache.get("_date") == today_str and len(cache) > 1:
        earnings = {k: v for k, v in cache.items() if k != "_date"}
        log.info(f"Earnings cache hit ({today_str}): {len(earnings)} tickers with dates")
        return earnings

    # Cache is stale — refresh from yfinance
    log.info(f"Refreshing earnings dates from yfinance for {len(universe)} tickers...")
    try:
        earnings = _fetch_yfinance_earnings(universe)
        # Save cache with date marker
        cache_data = {"_date": today_str}
        cache_data.update(earnings)
        with open(EARNINGS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        log.info(f"Earnings cache refreshed: {len(earnings)} tickers with upcoming dates")
        return earnings
    except Exception as e:
        log.warning(f"yfinance earnings fetch failed: {e}. Falling back to parquet cache.")

    # Fallback: static parquet cache (legacy)
    return _load_earnings_parquet_fallback()


def _load_earnings_parquet_fallback():
    """Legacy fallback: load earnings from parquet file."""
    cache_path = ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet"
    if not cache_path.exists():
        log.warning("No cached earnings dates found (neither yfinance nor parquet)")
        return {}

    df = pd.read_parquet(cache_path)
    result = {}
    date_col = "earnings_date" if "earnings_date" in df.columns else "date"
    today = pd.Timestamp.now()
    for ticker, grp in df.groupby("ticker"):
        dates = pd.to_datetime(grp[date_col]).sort_values()
        future = dates[dates >= today - pd.Timedelta(days=1)]
        if len(future) > 0:
            result[ticker] = future.iloc[0].strftime("%Y-%m-%d")

    log.info(f"Earnings parquet fallback loaded: {len(result)} tickers")
    return result


def has_earnings_soon(ticker, earnings_lookup, buffer_days=EARNINGS_BUFFER_DAYS):
    """Check if ticker has earnings within the next buffer_days calendar days.

    Returns (True, earnings_date_str) if earnings are within window,
    (False, None) otherwise.

    If ticker is not in earnings_lookup, assume no earnings (safe to trade).
    """
    if ticker not in earnings_lookup:
        return False, None

    try:
        earnings_date = pd.Timestamp(earnings_lookup[ticker])
        today = pd.Timestamp.now().normalize()
        days_until = (earnings_date - today).days

        if 0 <= days_until <= buffer_days:
            return True, earnings_date.strftime("%Y-%m-%d")
    except Exception:
        pass

    return False, None


# ── Real Option Chain Pricing ──
def fetch_option_chain(ticker_symbol, target_dte=DTE_TARGET):
    """Fetch real option chain from yfinance and find best expiry near target DTE.

    Returns (puts_df, expiry_str) or (None, None) on failure.
    puts_df has columns: strike, bid, ask, lastPrice, impliedVolatility, etc.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker_symbol)
        available = t.options  # tuple of 'YYYY-MM-DD' strings
        if not available:
            return None, None

        # Find expiry closest to target DTE
        today = datetime.utcnow().date()
        best_expiry = None
        best_dist = 9999
        for exp_str in available:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < DTE_MIN:
                continue
            dist = abs(dte - target_dte)
            if dist < best_dist:
                best_dist = dist
                best_expiry = exp_str

        if best_expiry is None:
            return None, None

        chain = t.option_chain(best_expiry)
        puts = chain.puts
        if puts is None or puts.empty:
            return None, None

        return puts, best_expiry
    except Exception as e:
        log.warning(f"Option chain fetch failed for {ticker_symbol}: {e}")
        return None, None


def find_spread_from_chain(puts_df, S, sigma, T, delta_target, spread_width):
    """Find short and long put legs from real option chain.

    Strategy:
    - Short leg: find put closest to target delta (using chain's IV or BS approximation)
    - Long leg: find put with strike closest to (short_strike - spread_width)

    Returns dict with leg details or None if no viable spread found.
    """
    if puts_df is None or puts_df.empty or S <= 0 or T <= 0:
        return None

    # Calculate approximate delta for each strike in the chain
    # Use chain's impliedVolatility if available, else fall back to historical sigma
    candidates = []
    for _, row in puts_df.iterrows():
        K = row['strike']
        if K <= 0 or K > S:  # Only OTM puts (strike below current price)
            if K >= S:
                continue
        iv = row.get('impliedVolatility', sigma)
        if iv is None or iv <= 0 or np.isnan(iv):
            iv = sigma

        # Compute BS delta for this strike
        try:
            d1 = (math.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * math.sqrt(T) + 1e-9)
            delta_abs = _Phi(-d1)
        except Exception:
            continue

        bid = row.get('bid', 0)
        ask = row.get('ask', 0)
        # Handle NaN values from yfinance
        if bid is None or (isinstance(bid, float) and np.isnan(bid)):
            bid = 0.0
        if ask is None or (isinstance(ask, float) and np.isnan(ask)):
            ask = 0.0
        bid = float(bid)
        ask = float(ask)
        last = row.get('lastPrice', 0)
        if last is None or (isinstance(last, float) and np.isnan(last)):
            last = 0.0
        mid = (bid + ask) / 2 if (bid + ask) > 0 else float(last)

        candidates.append({
            'strike': K,
            'delta': delta_abs,
            'bid': float(bid),
            'ask': float(ask),
            'mid': float(mid),
            'iv': float(iv),
        })

    if not candidates:
        return None

    # Find short leg: closest to target delta
    candidates.sort(key=lambda c: abs(c['delta'] - delta_target))
    short_leg = candidates[0]

    # Find long leg: strike closest to (short_strike - spread_width)
    target_long_strike = short_leg['strike'] - spread_width
    long_candidates = [c for c in candidates if c['strike'] < short_leg['strike']]
    if not long_candidates:
        return None

    long_candidates.sort(key=lambda c: abs(c['strike'] - target_long_strike))
    long_leg = long_candidates[0]

    # Validate: short bid > 0, we need to actually collect premium
    if short_leg['bid'] <= 0:
        return None

    # Net credit = short_bid - long_ask (realistic fill)
    net_credit_per_share = short_leg['bid'] - long_leg['ask']
    actual_width = short_leg['strike'] - long_leg['strike']

    if actual_width <= 0:
        return None

    # Calculate effective BA% for each leg
    short_ba_pct = ((short_leg['ask'] - short_leg['bid']) / short_leg['mid'] * 100
                    if short_leg['mid'] > 0 else 0)
    long_ba_pct = ((long_leg['ask'] - long_leg['bid']) / long_leg['mid'] * 100
                   if long_leg['mid'] > 0 else 0)

    return {
        'short_strike': short_leg['strike'],
        'short_bid': short_leg['bid'],
        'short_ask': short_leg['ask'],
        'short_mid': short_leg['mid'],
        'short_delta': short_leg['delta'],
        'short_iv': short_leg['iv'],
        'short_ba_pct': short_ba_pct,
        'long_strike': long_leg['strike'],
        'long_bid': long_leg['bid'],
        'long_ask': long_leg['ask'],
        'long_mid': long_leg['mid'],
        'long_delta': long_leg['delta'],
        'long_iv': long_leg['iv'],
        'long_ba_pct': long_ba_pct,
        'net_credit_per_share': net_credit_per_share,
        'actual_width': actual_width,
        'pricing_source': 'REAL_CHAIN',
    }


def log_fill_quality(ticker, spread_info, net_credit_bs_per_share):
    """Log fill quality comparison between real chain and BS pricing."""
    try:
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'ticker': ticker,
            'short_strike': spread_info['short_strike'],
            'long_strike': spread_info['long_strike'],
            'short_bid': spread_info['short_bid'],
            'short_ask': spread_info['short_ask'],
            'short_mid': spread_info['short_mid'],
            'short_ba_pct': round(spread_info['short_ba_pct'], 1),
            'long_bid': spread_info['long_bid'],
            'long_ask': spread_info['long_ask'],
            'long_mid': spread_info['long_mid'],
            'long_ba_pct': round(spread_info['long_ba_pct'], 1),
            'net_credit_real': round(spread_info['net_credit_per_share'], 4),
            'net_credit_bs': round(net_credit_bs_per_share, 4),
            'effective_ba_pct': round(
                (1 - spread_info['net_credit_per_share'] /
                 ((spread_info['short_mid'] - spread_info['long_mid']) or 1e-9)) * 100, 1),
            'pricing_source': spread_info['pricing_source'],
        }
        with open(FILL_QUALITY_FILE, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    except Exception as e:
        log.warning(f"Failed to log fill quality: {e}")


# ── Black-Scholes ──
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def _get_option_price_consistent(ticker, S, K, T, sigma, expiry_date, kind='put'):
    """Get option price using SAME source as entry (Alpaca if available, else BS).
    HC #709 audit fix: entry used Alpaca pricing (IV 0.35-0.47) but close used BS (IV 0.21-0.28),
    creating instant phantom profit on every position. Now both use the same source."""
    if _HAS_PRICING_BRIDGE:
        try:
            price = _alpaca_get_premium(
                ticker, S, K, T, sigma, expiry_date,
                kind=kind, bs_price_fn=bs_price)
            if price and price > 0:
                return price
        except Exception:
            pass
    return bs_price(S, K, T, sigma, kind=kind)


def find_strike(S, sigma, T, delta_target, kind="put"):
    """Binary search for strike at target delta."""
    if T <= 0 or sigma <= 0:
        return S
    if kind == "put":
        lo, hi = S * 0.3, S * 1.0
    else:
        lo, hi = S * 1.0, S * 2.0
    for _ in range(60):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            delta_abs = _Phi(-d1)
        else:
            delta_abs = _Phi(d1)
        if delta_abs > delta_target:
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            if kind == "put":
                lo = K
            else:
                hi = K
    return round(K * 2) / 2  # Round to nearest $0.50


def apply_slippage(premium):
    """Apply slippage to a premium (reduces credit received, increases debit paid)."""
    return max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)


# ── Price Data ──
def get_live_prices(tickers, batch_size=50):
    """Fetch current prices and 20d realized vol via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="60d", auto_adjust=True,
                             threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data["Close"].columns:
                        close = data["Close"][t].dropna()
                        if len(close) > 0:
                            prices[t] = float(close.iloc[-1])
                            log_ret = np.log(close / close.shift(1)).dropna()
                            if len(log_ret) >= 20:
                                sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252))
                            elif len(log_ret) > 5:
                                sigmas[t] = float(log_ret.std() * np.sqrt(252))
                            sigmas[t] = max(0.05, min(sigmas.get(t, 0.3), 2.0))
            elif len(batch) == 1 and not data.empty:
                t = batch[0]
                close = data["Close"].dropna()
                if len(close) > 0:
                    prices[t] = float(close.iloc[-1])
                    log_ret = np.log(close / close.shift(1)).dropna()
                    sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else 0.3
                    sigmas[t] = max(0.05, min(sigmas[t], 2.0))
        except Exception as e:
            log.warning(f"Price batch {i}-{i+batch_size} failed: {e}")
        time.sleep(0.5)

    return prices, sigmas


def get_vix():
    """Fetch current VIX."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period='5d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return 20.0


def vix_size_scalar(vix: float) -> float:
    """Return position size scalar based on current VIX level."""
    if not VIX_SCALE_ENABLED:
        return 1.0
    for threshold, scale in VIX_SCALE_TIERS:
        if vix < threshold:
            return scale
    return 0.0


def get_dynamic_delta(vix: float) -> float:
    """Return put delta target based on VIX regime.

    - VIX < 15:  delta=0.35 (more aggressive — capture richer premiums in calm markets)
    - VIX 15-22: delta=0.30 (standard)
    - VIX > 22:  delta=0.25 (more conservative — reduce breach risk in volatile markets)

    Falls back to 0.30 if VIX is unavailable (caller passes default 20.0).
    """
    if vix < 15:
        delta = 0.35
    elif vix <= 22:
        delta = 0.30
    else:
        delta = 0.25
    log.info(f"DYNAMIC DELTA: VIX={vix:.1f} -> delta={delta}")
    return delta


def is_bear_regime():
    """SPY below 50d SMA = bear regime."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period='70d')
        if len(hist) >= 50:
            sma50 = hist['Close'].tail(50).mean()
            current = float(hist['Close'].iloc[-1])
            return current < sma50
    except Exception as e:
        log.warning(f"Regime check failed: {e}")
    return False


# ── Equity Curve Brake ──
def load_nav_history():
    if NAV_HISTORY_FILE.exists():
        with open(NAV_HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_nav_history(history):
    with open(NAV_HISTORY_FILE, 'w') as f:
        json.dump(history, f)


def compute_brake_scale(nav_history, current_nav):
    """
    Equity curve brake: if NAV is below (1-threshold) of 60-day peak,
    scale exposure to 25%.
    """
    if len(nav_history) < 5:
        return 1.0

    recent = nav_history[-BRAKE_LOOKBACK_DAYS:]
    peak = max(entry["nav"] for entry in recent)

    if current_nav < peak * (1 - BRAKE_THRESHOLD):
        log.info(f"EQUITY BRAKE ACTIVE: NAV ${current_nav:,.0f} < peak ${peak:,.0f} x "
                 f"{1-BRAKE_THRESHOLD:.0%} = ${peak*(1-BRAKE_THRESHOLD):,.0f}. "
                 f"Scale -> {BRAKE_SCALE:.0%}")
        return BRAKE_SCALE

    return 1.0


def check_dd_trigger(nav_history):
    """
    Fast drawdown trigger (HC #662 R4).
    Returns True if trailing 3-day return < -5%, meaning we should halt new entries.

    Validated by permutation test (p=0.000): bad days cluster 2.3x, so when
    recent returns are bad, more bad days are statistically likely to follow.
    Backtest improvement: Sharpe 4.09→4.26, MaxDD -30%→-22%.
    """
    if not DD_TRIGGER_ENABLED:
        return False

    if len(nav_history) < DD_TRIGGER_LOOKBACK + 1:
        return False

    recent = nav_history[-(DD_TRIGGER_LOOKBACK + 1):]
    nav_start = recent[0]["nav"]
    nav_end = recent[-1]["nav"]

    if nav_start <= 0:
        return False

    trail_return = (nav_end - nav_start) / nav_start

    if trail_return < DD_TRIGGER_THRESHOLD:
        log.info(f"DD TRIGGER ACTIVE: {DD_TRIGGER_LOOKBACK}d return = "
                 f"{trail_return*100:.1f}% < {DD_TRIGGER_THRESHOLD*100:.0f}% threshold. "
                 f"Halting new entries.")
        return True

    return False


# ── Circuit Breaker ──
def load_circuit_breaker():
    """Load circuit breaker state."""
    if CIRCUIT_BREAKER_FILE.exists():
        try:
            with open(CIRCUIT_BREAKER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"frozen_until": None, "trigger_date": None, "trigger_loss_pct": None}


def save_circuit_breaker(cb_state):
    """Save circuit breaker state."""
    with open(CIRCUIT_BREAKER_FILE, 'w') as f:
        json.dump(cb_state, f, indent=2, default=str)


def check_circuit_breaker(nav: float) -> bool:
    """
    Circuit breaker: if daily P&L loss > 2% of portfolio, freeze new entries for 1 day.
    Returns True if frozen (should skip new entries).
    """
    if not CIRCUIT_BREAKER_ENABLED:
        return False

    cb_state = load_circuit_breaker()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Check if we're currently frozen
    if cb_state.get("frozen_until"):
        frozen_until = cb_state["frozen_until"]
        if today <= frozen_until:
            log.info(f"CIRCUIT BREAKER FROZEN until {frozen_until} "
                     f"(triggered {cb_state.get('trigger_date')} at "
                     f"{cb_state.get('trigger_loss_pct', 0):.1%} loss)")
            return True
        else:
            # Freeze expired, clear it
            log.info(f"Circuit breaker freeze expired (was until {frozen_until})")
            cb_state = {"frozen_until": None, "trigger_date": None, "trigger_loss_pct": None}
            save_circuit_breaker(cb_state)

    # Check today's P&L against SOD NAV
    sod_data = {}
    if SOD_NAV_FILE.exists():
        try:
            sod_data = json.loads(SOD_NAV_FILE.read_text())
        except Exception:
            pass

    if sod_data.get("date") != today or sod_data.get("sod_nav", 0) <= 0:
        return False

    sod_nav = sod_data["sod_nav"]
    daily_loss_pct = (sod_nav - nav) / sod_nav

    if daily_loss_pct >= CIRCUIT_BREAKER_LOSS_PCT:
        # Trigger circuit breaker
        freeze_until = (datetime.utcnow() + timedelta(days=CIRCUIT_BREAKER_FREEZE_DAYS)).strftime("%Y-%m-%d")
        cb_state = {
            "frozen_until": freeze_until,
            "trigger_date": today,
            "trigger_loss_pct": round(daily_loss_pct, 4),
            "sod_nav": round(sod_nav, 2),
            "trigger_nav": round(nav, 2),
        }
        save_circuit_breaker(cb_state)
        log.warning(f"CIRCUIT BREAKER TRIGGERED: daily loss {daily_loss_pct:.1%} "
                    f"(>${CIRCUIT_BREAKER_LOSS_PCT:.0%} threshold). "
                    f"SOD NAV ${sod_nav:,.0f} -> ${nav:,.0f}. "
                    f"Freezing new entries until {freeze_until}.")
        return True

    return False


# ── State Management ──
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': STARTING_CAPITAL,
        'spreads': [],       # list of open spread positions
        'realized_pnl': 0.0,
        'trade_count': 0,
        'start_date': datetime.utcnow().isoformat(),
        'last_check': None,
        'stop_loss_cooldown': {},  # {ticker: ISO timestamp of last stop-loss exit}
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def compute_nav(state, prices, sigmas):
    """Compute current NAV = cash + unrealized spread MTM."""
    nav = state['cash']

    for sp in state['spreads']:
        ticker = sp['ticker']
        if ticker not in prices:
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, sp.get('sigma', 0.20))
        expiry = pd.Timestamp(sp['expiry'])
        T = max((expiry - pd.Timestamp.now()).days, 0) / 365.0

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        short_val = _get_option_price_consistent(ticker, S, sp['short_strike'], T, sigma, expiry_date, kind='put')
        long_val = _get_option_price_consistent(ticker, S, sp['long_strike'], T, sigma, expiry_date, kind='put')
        spread_mtm = (short_val - long_val) * 100 * sp.get('contracts', 1)

        # We owe the spread value (short spread = liability)
        nav -= spread_mtm

    return nav


SOD_NAV_FILE = STATE_DIR / "sod_nav.json"
NAV_DROP_ALERT_PCT = 0.025
_nav_drop_alerted_today = set()


def check_nav_drop(nav: float) -> None:
    """Alert if NAV drops >2.5% from start-of-day."""
    today = datetime.utcnow().strftime("%Y-%m-%d")

    sod_data = {}
    if SOD_NAV_FILE.exists():
        try:
            sod_data = json.loads(SOD_NAV_FILE.read_text())
        except Exception:
            pass

    if sod_data.get("date") != today:
        sod_data = {"date": today, "sod_nav": round(nav, 2)}
        SOD_NAV_FILE.write_text(json.dumps(sod_data))
        _nav_drop_alerted_today.clear()
        return

    sod_nav = sod_data["sod_nav"]
    if sod_nav <= 0:
        return

    drop_pct = (sod_nav - nav) / sod_nav
    if drop_pct >= NAV_DROP_ALERT_PCT and today not in _nav_drop_alerted_today:
        _nav_drop_alerted_today.add(today)
        log.warning(f"NAV DROP ALERT: {drop_pct:.1%} intraday "
                    f"(SOD: ${sod_nav:,.0f} -> ${nav:,.0f})")
        try:
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"WHEEL BPS NAV DROP: {drop_pct:.1%} intraday '
                f'(${sod_nav:,.0f} -> ${nav:,.0f})" 2>/dev/null'
            )
        except Exception:
            pass


def log_equity(state, nav):
    row = f"{datetime.utcnow().isoformat()},{nav:.2f},{state['realized_pnl']:.2f}\n"
    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, 'w') as f:
            f.write("timestamp,nav,realized_pnl\n")
    with open(EQUITY_FILE, 'a') as f:
        f.write(row)


def log_trade(info):
    with open(TRADES_FILE, 'a') as f:
        f.write(json.dumps(info, default=str) + '\n')


# ── Expiry Logic (weekly) ──
def find_expiry(from_date=None):
    """Find the nearest Friday within DTE_MIN..DTE_MAX of target DTE."""
    if from_date is None:
        from_date = datetime.utcnow()
    best, best_dist = None, 10000
    for d_off in range(DTE_MIN, DTE_MAX + 1):
        cand = from_date + timedelta(days=d_off)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + timedelta(days=shift)
        dte = (cand_fri - from_date).days
        if dte < DTE_MIN or dte > DTE_MAX:
            continue
        dist = abs(dte - DTE_TARGET)
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


# ── Trading Logic ──
def process_spreads(state, prices, sigmas):
    """Check existing spreads for profit-take, expiry, or max-loss."""
    now = pd.Timestamp.now()
    new_spreads = []

    for sp in state['spreads']:
        ticker = sp['ticker']
        if ticker not in prices:
            new_spreads.append(sp)
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, sp.get('sigma', 0.20))
        expiry = pd.Timestamp(sp['expiry'])
        T_days = (expiry - now).days
        T = max(T_days, 0) / 365.0
        contracts = sp.get('contracts', 1)

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        short_val = _get_option_price_consistent(ticker, S, sp['short_strike'], T, sigma, expiry_date, kind='put')
        long_val = _get_option_price_consistent(ticker, S, sp['long_strike'], T, sigma, expiry_date, kind='put')
        spread_val_per_share = short_val - long_val  # positive = we owe this

        # Cost to close both legs (buy back short, sell long)
        close_cost_short = COST_PER_CONTRACT * contracts  # commission to buy back short
        close_cost_long = COST_PER_CONTRACT * contracts    # commission to sell long
        total_close_cost = spread_val_per_share * 100 * contracts + close_cost_short + close_cost_long

        # Premium received at open (already in cash)
        premium_received = sp['premium_received']

        # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
        entry_dt = pd.Timestamp(sp.get('entry_date', '2020-01-01'))
        hours_held = (now - entry_dt).total_seconds() / 3600
        if hours_held < 24:
            new_spreads.append(sp)
            continue

        # ── Profit take at 65% ──
        if premium_received > 0:
            profit_pct = (premium_received - total_close_cost) / premium_received
            if profit_pct >= PROFIT_TAKE_PCT:
                # Close spread: buy back short put, sell long put
                state['cash'] -= total_close_cost
                pnl = premium_received - total_close_cost
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"PROFIT TAKE BPS {ticker} @ {profit_pct:.0%}. "
                         f"PnL: ${pnl:.2f} ({contracts} contracts)")
                log_trade({
                    'action': 'profit_take', 'ticker': ticker,
                    'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                    'pnl': round(pnl, 2), 'contracts': contracts,
                    'time': now.isoformat(),
                })
                continue

            # Stop-loss: close spread when loss reaches 1x premium collected
            if profit_pct <= -BPS_STOP_LOSS_MULT:
                state['cash'] -= total_close_cost
                pnl = premium_received - total_close_cost
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"STOP LOSS BPS {ticker} @ {profit_pct:.0%} (loss >= {BPS_STOP_LOSS_MULT:.0f}x premium). "
                         f"PnL: ${pnl:.2f}")
                log_trade({
                    'action': 'stop_loss_bps', 'ticker': ticker,
                    'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                    'pnl': round(pnl, 2), 'contracts': contracts,
                    'profit_pct': round(profit_pct, 3), 'time': now.isoformat(),
                })
                # Record stop-loss cooldown — block re-entry for 24 hours
                if 'stop_loss_cooldown' not in state:
                    state['stop_loss_cooldown'] = {}
                state['stop_loss_cooldown'][ticker] = now.isoformat()
                log.info(f"COOLDOWN SET: {ticker} blocked for {STOP_LOSS_COOLDOWN_HOURS}h after stop-loss")
                continue

        # ── Close 1 DTE (assignment risk reduction — HC #664 study found +10% Sharpe) ──
        if 0 < T_days <= 1 and now < expiry:
            # Buy back the spread to avoid pin risk and overnight gap risk
            state['cash'] -= total_close_cost
            pnl = premium_received - total_close_cost
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"CLOSE 1DTE BPS {ticker} "
                     f"{sp['short_strike']:.2f}/{sp['long_strike']:.2f} "
                     f"PnL: ${pnl:.2f} ({contracts}c) "
                     f"[pin risk avoidance]")
            log_trade({
                'action': 'close_1dte', 'ticker': ticker,
                'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'underlying_price': round(S, 2),
                'time': now.isoformat(),
            })
            continue

        # ── Expiry ──
        if now >= expiry:
            short_itm = S < sp['short_strike']
            long_itm = S < sp['long_strike']
            close_fees = 2 * COST_PER_CONTRACT * contracts

            if not short_itm:
                # Both expire OTM -- full premium kept
                pnl = premium_received - close_fees
                state['cash'] -= close_fees
            elif short_itm and not long_itm:
                # Short put ITM, long put OTM -- partial loss
                intrinsic_loss = (sp['short_strike'] - S) * 100 * contracts
                state['cash'] -= intrinsic_loss + close_fees
                pnl = premium_received - intrinsic_loss - close_fees
            else:
                # Both ITM -- max loss = spread width
                max_loss = (sp['short_strike'] - sp['long_strike']) * 100 * contracts
                state['cash'] -= max_loss + close_fees
                pnl = premium_received - max_loss - close_fees

            state['realized_pnl'] += pnl
            state['trade_count'] += 1

            outcome = "OTM" if not short_itm else ("PARTIAL" if not long_itm else "MAX_LOSS")
            log.info(f"EXPIRED {outcome} BPS {ticker} "
                     f"{sp['short_strike']:.2f}/{sp['long_strike']:.2f}. "
                     f"PnL: ${pnl:.2f}")
            log_trade({
                'action': f'expired_{outcome.lower()}', 'ticker': ticker,
                'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'underlying_at_expiry': round(S, 2),
                'time': now.isoformat(),
            })
            continue

        # ── Max loss early exit: underlying below long strike ──
        # If underlying is well below long strike, spread is at max loss.
        # Close early to free up margin (no point holding).
        if S < sp['long_strike'] * 0.98:  # 2% buffer below long strike
            max_loss_value = (sp['short_strike'] - sp['long_strike']) * 100 * contracts
            close_fees = 2 * COST_PER_CONTRACT * contracts
            state['cash'] -= max_loss_value + close_fees
            pnl = premium_received - max_loss_value - close_fees
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            log.info(f"EARLY CLOSE (max loss) BPS {ticker} @ ${S:.2f} "
                     f"(below long strike {sp['long_strike']:.2f}). PnL: ${pnl:.2f}")
            log_trade({
                'action': 'early_close_max_loss', 'ticker': ticker,
                'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'underlying_price': round(S, 2),
                'time': now.isoformat(),
            })
            continue

        # Position survives -- keep it
        new_spreads.append(sp)

    state['spreads'] = new_spreads


def open_new_spreads(state, prices, sigmas, vix, bear_mode, brake_scale,
                     earnings_lookup, universe, dd_trigger_active=False,
                     circuit_breaker_active=False):
    """Open new BPS positions on tickers without existing positions."""
    if bear_mode:
        log.info("BEAR REGIME -- no new spreads")
        return
    if circuit_breaker_active:
        log.info("CIRCUIT BREAKER ACTIVE -- 2% daily loss freeze, no new entries")
        return
    # VIX-scaled sizing (replaces old VIX_MAX_GATE binary check)
    vix_scale = vix_size_scalar(vix)
    if vix_scale <= 0:
        log.info(f"VIX {vix:.1f} -- VIX scale = 0 (fully gated, no new entries)")
        return
    elif vix_scale < 1.0:
        log.info(f"VIX {vix:.1f} -- VIX scale = {vix_scale:.0%} (reduced sizing)")
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT -- no new spreads")
        return
    if dd_trigger_active:
        log.info("DD TRIGGER HALT -- 3-day trailing return below threshold, no new entries")
        return

    active_tickers = {sp['ticker'] for sp in state['spreads']}

    # Cross-engine dedup: avoid tickers already held by standard BPS engine
    try:
        import json as _json
        _peer_state_path = Path(__file__).parent / 'wheel_bps_state' / 'state.json'
        if _peer_state_path.exists():
            with open(_peer_state_path) as _f:
                _peer = _json.load(_f)
            _peer_tickers = {sp['ticker'] for sp in _peer.get('spreads', [])}
            active_tickers |= _peer_tickers
    except Exception:
        pass  # If peer state can't be read, proceed without dedup

    nav = compute_nav(state, prices, sigmas)

    # Scale margin cap by brake AND VIX scalar
    effective_margin_cap = MARGIN_CAP * brake_scale * vix_scale
    current_margin = sum(
        (sp['short_strike'] - sp['long_strike']) * 100 * sp.get('contracts', 1)
        for sp in state['spreads']
    )
    available_margin = nav * effective_margin_cap - current_margin
    per_name_limit = nav * PER_NAME_PCT * brake_scale * vix_scale

    if available_margin <= 0:
        return

    # Check max concurrent
    if len(state['spreads']) >= MAX_CONCURRENT:
        return

    # Sort by IV (sigma) descending — higher IV = more premium = better spread
    # Matches backtest which sorted by IV rank. Random shuffle was a deviation.
    import random
    candidates = [t for t in universe if t in prices and t not in active_tickers]
    candidates.sort(key=lambda t: sigmas.get(t, 0), reverse=True)

    # Dynamic delta based on VIX regime (replaces fixed PUT_DELTA_TARGET)
    delta_target = get_dynamic_delta(vix)

    opened = 0
    for ticker in candidates:
        # Check position cap at TOP of loop (fixes bug where batch exceeds MAX_CONCURRENT)
        if len(state['spreads']) >= MAX_CONCURRENT:
            break
        if ticker not in prices or ticker not in sigmas:
            continue

        S = prices[ticker]
        sigma = sigmas[ticker]

        # Price filter
        if S < MIN_PRICE or S > MAX_PRICE:
            continue

        # Blacklist: consistent losers across engines
        if ticker in LOSER_BLACKLIST:
            continue

        # Sigma filter: skip extreme-vol names (sigma > 0.8 annualized)
        # IBM (sigma=1.1), ARM, FSLR, APD all triggered repeated stop-losses.
        if sigma > SIGMA_MAX_ENTRY:
            log.info(f"SIGMA SKIP: {ticker} sigma={sigma:.2f} > {SIGMA_MAX_ENTRY:.2f} threshold")
            continue

        # Stop-loss cooldown: skip ticker if stopped out within last 24 hours
        _cooldown_map = state.get('stop_loss_cooldown', {})
        if ticker in _cooldown_map:
            try:
                _last_stop = pd.Timestamp(_cooldown_map[ticker])
                _hours_elapsed = (pd.Timestamp.now() - _last_stop).total_seconds() / 3600
                if _hours_elapsed < STOP_LOSS_COOLDOWN_HOURS:
                    log.info(f"COOLDOWN SKIP: {ticker} stopped out {_hours_elapsed:.1f}h ago "
                             f"(cooldown: {STOP_LOSS_COOLDOWN_HOURS}h)")
                    continue
                else:
                    # Cooldown expired — clean up
                    del state['stop_loss_cooldown'][ticker]
            except Exception:
                pass

        # Earnings filter (7-day buffer — HC #664)
        earnings_hit, earnings_date = has_earnings_soon(ticker, earnings_lookup)
        if earnings_hit:
            log.info(f"EARNINGS SKIP: {ticker} earnings on {earnings_date}")
            continue

        # ── Try REAL option chain pricing first ──
        pricing_source = 'BS_FALLBACK'
        spread_info = None
        K_short = None
        K_long = None
        net_credit_per_share = None
        actual_width = SPREAD_WIDTH
        net_credit_bs_per_share = None  # for fill quality comparison

        puts_df, chain_expiry = fetch_option_chain(ticker, target_dte=DTE_TARGET)
        time.sleep(0.5)  # Rate limit protection between chain fetches

        if puts_df is not None and chain_expiry is not None:
            # Use real chain expiry
            expiry_date = datetime.strptime(chain_expiry, "%Y-%m-%d")
            T = max((expiry_date - datetime.utcnow()).days, 0) / 365.0
            dte = (expiry_date - datetime.utcnow()).days

            if T <= 0 or dte < DTE_MIN:
                continue

            # Try to find spread from real chain
            spread_info = find_spread_from_chain(puts_df, S, sigma, T, delta_target, SPREAD_WIDTH)

            if spread_info is not None:
                K_short = spread_info['short_strike']
                K_long = spread_info['long_strike']
                net_credit_per_share = spread_info['net_credit_per_share']
                actual_width = spread_info['actual_width']
                pricing_source = 'REAL_CHAIN'

                # Also compute BS price for comparison logging
                prem_short_bs = bs_price(S, K_short, T, sigma, kind='put')
                prem_long_bs = bs_price(S, K_long, T, sigma, kind='put')
                net_credit_bs_per_share = prem_short_bs - prem_long_bs

                log.info(f"REAL CHAIN {ticker}: short {K_short} bid={spread_info['short_bid']:.2f} "
                         f"ask={spread_info['short_ask']:.2f} BA={spread_info['short_ba_pct']:.0f}%, "
                         f"long {K_long} bid={spread_info['long_bid']:.2f} "
                         f"ask={spread_info['long_ask']:.2f} BA={spread_info['long_ba_pct']:.0f}%, "
                         f"net_credit_real={net_credit_per_share:.3f} "
                         f"net_credit_bs={net_credit_bs_per_share:.3f}")
            else:
                log.info(f"REAL CHAIN {ticker}: no viable spread found, falling back to BS")

        # ── ALPACA/BS FALLBACK: if yfinance chain failed or no viable spread ──
        if pricing_source == 'BS_FALLBACK':
            expiry_date = find_expiry()
            if expiry_date is None:
                continue
            T = (expiry_date - datetime.utcnow()).days / 365.0
            dte = (expiry_date - datetime.utcnow()).days

            K_short = find_strike(S, sigma, T, delta_target, kind='put')
            K_long = K_short - SPREAD_WIDTH
            actual_width = SPREAD_WIDTH

            if K_long <= 0 or K_short <= 0:
                continue

            # Try Alpaca real pricing before falling back to BS
            _exp_d = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
            if _HAS_PRICING_BRIDGE:
                prem_short = _alpaca_get_premium(ticker, S, K_short, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
                prem_long = _alpaca_get_premium(ticker, S, K_long, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
                log.info(f"ALPACA/BS FALLBACK: {ticker}")
            else:
                prem_short = bs_price(S, K_short, T, sigma, kind='put')
                prem_long = bs_price(S, K_long, T, sigma, kind='put')
                log.info(f"BS_FALLBACK: {ticker}")
            net_prem_bs = prem_short - prem_long

            if net_prem_bs < MIN_NET_CREDIT:
                continue

            # Apply slippage to BS pricing (old cost model)
            prem_short_after_slip = apply_slippage(prem_short)
            prem_long_after_slip = prem_long * (1 + SLIPPAGE_FRAC)
            net_credit_per_share = prem_short_after_slip - prem_long_after_slip
            net_credit_bs_per_share = net_prem_bs

        # ── Common path: sizing and execution ──
        if net_credit_per_share is None or net_credit_per_share <= 0:
            continue

        if net_credit_per_share < MIN_NET_CREDIT:
            if pricing_source == 'REAL_CHAIN':
                log.info(f"SKIP {ticker}: real net credit ${net_credit_per_share:.3f} "
                         f"< min ${MIN_NET_CREDIT:.2f}")
            continue

        # Margin per contract = actual spread width x 100
        margin_per_contract = actual_width * 100

        # Position sizing: how many contracts?
        max_by_name = int(per_name_limit // margin_per_contract)
        max_by_margin = int(available_margin // margin_per_contract)
        n_contracts = min(max_by_name, max_by_margin)
        n_contracts = max(n_contracts, 1)

        # Re-check margin fits
        total_margin = margin_per_contract * n_contracts
        if total_margin > available_margin:
            n_contracts = int(available_margin // margin_per_contract)
        if n_contracts < 1:
            continue
        total_margin = margin_per_contract * n_contracts

        net_credit = net_credit_per_share * 100 * n_contracts
        # Commission: $0.65/contract per leg, 2 legs
        open_commission = 2 * COST_PER_CONTRACT * n_contracts
        net_credit -= open_commission

        if net_credit <= 0:
            continue

        # Log fill quality for real chain trades
        if spread_info is not None and net_credit_bs_per_share is not None:
            log_fill_quality(ticker, spread_info, net_credit_bs_per_share)

        # Execute: add net credit to cash (margin is held implicitly)
        state['cash'] += net_credit

        state['spreads'].append({
            'ticker': ticker,
            'short_strike': K_short,
            'long_strike': K_long,
            'contracts': n_contracts,
            'premium_received': net_credit,  # total $ received after costs
            'sigma': sigma,
            'expiry': expiry_date.isoformat(),
            'open_date': datetime.utcnow().isoformat(),
            'spread_width': actual_width,
            'margin_held': total_margin,
            'pricing_source': pricing_source,
        })

        available_margin -= total_margin
        opened += 1

        source_tag = "[REAL]" if pricing_source == 'REAL_CHAIN' else "[BS]"
        log.info(f"SELL BPS {source_tag} {ticker} {K_short:.2f}/{K_long:.2f} "
                 f"({dte}d, {n_contracts}c), "
                 f"credit ${net_credit:.2f}, "
                 f"yield {net_credit/total_margin*100:.1f}%")
        log_trade({
            'action': 'sell_bps', 'ticker': ticker,
            'short_strike': K_short, 'long_strike': K_long,
            'contracts': n_contracts, 'net_credit': round(net_credit, 2),
            'dte': dte, 'sigma': round(sigma, 3),
            'underlying_price': round(S, 2),
            'spread_width': actual_width,
            'brake_scale': brake_scale,
            'pricing_source': pricing_source,
            'time': datetime.utcnow().isoformat(),
        })

        if len(state['spreads']) >= MAX_CONCURRENT:
            break

    if opened:
        log.info(f"Opened {opened} new BPS (brake scale: {brake_scale:.0%}, "
                 f"total spreads: {len(state['spreads'])})")


# ── Main Loop ──
def run_cycle(universe, earnings_lookup=None):
    """Run one full check cycle.

    earnings_lookup is refreshed daily via the cache mechanism in
    load_earnings_dates_yf, so we call it each cycle (cache hit is free).
    """
    # Refresh earnings cache (daily — loads from file if already fetched today)
    earnings_lookup = load_earnings_dates_yf(universe)

    state = load_state()

    # Fetch prices (positions + subset of universe)
    position_tickers = {sp['ticker'] for sp in state['spreads']}
    needed = list(position_tickers | set(universe[:100]))
    prices, sigmas = get_live_prices(needed)
    vix = get_vix()
    bear_mode = is_bear_regime()

    nav = compute_nav(state, prices, sigmas)
    total_margin = sum(
        (sp['short_strike'] - sp['long_strike']) * 100 * sp.get('contracts', 1)
        for sp in state['spreads']
    )
    margin_util = total_margin / nav * 100 if nav > 0 else 0

    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"spreads={len(state['spreads'])}, margin={margin_util:.1f}%, "
             f"VIX={vix:.1f}, regime={'BEAR' if bear_mode else 'BULL'}")

    # Update NAV history for brake
    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(), "nav": round(nav, 2)})
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    # Compute brake scale
    brake_scale = compute_brake_scale(nav_history, nav)

    # Fast drawdown trigger (HC #662 R4)
    dd_trigger_active = check_dd_trigger(nav_history)

    # Circuit breaker: 2% daily loss → freeze new entries for 1 calendar day
    circuit_breaker_active = check_circuit_breaker(nav)

    # Bear mode: close all spreads early (buy back)
    if bear_mode:
        for sp in list(state['spreads']):
            ticker = sp['ticker']
            if ticker not in prices:
                continue
            S = prices[ticker]
            sigma = sigmas.get(ticker, 0.20)
            expiry = pd.Timestamp(sp['expiry'])
            T = max((expiry - pd.Timestamp.now()).days, 0) / 365.0
            contracts = sp.get('contracts', 1)

            # FIX (HC #709 audit): Use consistent pricing — same source as entry
            expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
            short_val = _get_option_price_consistent(ticker, S, sp['short_strike'], T, sigma, expiry_date, kind='put')
            long_val = _get_option_price_consistent(ticker, S, sp['long_strike'], T, sigma, expiry_date, kind='put')
            close_cost = (short_val - long_val) * 100 * contracts
            close_fees = 2 * COST_PER_CONTRACT * contracts

            state['cash'] -= close_cost + close_fees
            pnl = sp['premium_received'] - close_cost - close_fees
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            state['spreads'].remove(sp)

            log.info(f"BEAR CLOSE BPS {ticker}. PnL: ${pnl:.2f}")
            log_trade({
                'action': 'bear_close', 'ticker': ticker,
                'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'time': pd.Timestamp.now().isoformat(),
            })

    # Process existing spreads
    process_spreads(state, prices, sigmas)

    # Open new spreads (skip if circuit breaker is active)
    open_new_spreads(state, prices, sigmas, vix, bear_mode, brake_scale,
                     earnings_lookup, universe, dd_trigger_active=dd_trigger_active,
                     circuit_breaker_active=circuit_breaker_active)

    # Final NAV
    nav = compute_nav(state, prices, sigmas)
    check_nav_drop(nav)
    log_equity(state, nav)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    dd_status = "HALTED" if dd_trigger_active else "normal"
    cb_status = "FROZEN" if circuit_breaker_active else "normal"
    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"spreads={len(state['spreads'])}, trades={state['trade_count']}, "
             f"dd_trigger={dd_status}, circuit_breaker={cb_status}")


def main():
    log.info("=" * 60)
    log.info("BULL PUT SPREAD (BPS) PAPER ENGINE")
    log.info(f"  Spread: ${SPREAD_WIDTH:.0f} wide, Delta: DYNAMIC (VIX<15:0.35, 15-22:0.30, >22:0.25)")
    log.info(f"  DTE: {DTE_TARGET}d (weekly), PT: {PROFIT_TAKE_PCT:.0%}")
    log.info(f"  Margin: {MARGIN_CAP:.0%} cap, {PER_NAME_PCT:.0%} per name")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg, REAL chain bid/ask (BS fallback: {SLIPPAGE_FRAC:.1%} slip)")
    log.info(f"  Optimal d30 config: dynamic delta (VIX-based), $15 wide, 25% margin, 65% PT, 15 max pos")
    log.info(f"  Circuit breaker: {CIRCUIT_BREAKER_LOSS_PCT:.0%} daily loss -> {CIRCUIT_BREAKER_FREEZE_DAYS}d freeze")
    log.info(f"  VIX hard cutoff: {VIX_MAX_GATE:.0f}")
    log.info(f"  Earnings buffer: {EARNINGS_BUFFER_DAYS}d (yfinance, daily cache)")
    log.info("=" * 60)

    # Initialize Alpaca real pricing (falls back to BS when unavailable)
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='wheel-bps-conservative')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    universe = load_universe()
    earnings_lookup = load_earnings_dates_yf(universe)

    log.info(f"Universe: {len(universe)} tickers, earnings data: {len(earnings_lookup)} tickers")

    while True:
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            # Market hours: Mon-Fri, 13:30-21:00 UTC (9:30 AM - 4 PM ET)
            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle(universe, earnings_lookup)
            else:
                if now.minute < 5:  # Once per hour off-hours
                    state = load_state()
                    log.info(f"Off-hours: spreads={len(state['spreads'])}, "
                             f"realized=${state['realized_pnl']:.2f}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == '__main__':
    main()
