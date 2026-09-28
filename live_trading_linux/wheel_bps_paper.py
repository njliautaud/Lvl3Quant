#!/usr/bin/env python3
"""
Bull Put Spread (BPS) Paper Engine — Winning Config from Higher-Returns Study
==============================================================================

Sells $10-wide bull put spreads on a 230-ticker equity universe.
This is the capital-efficient cousin of wheel_v4_paper.py (CSP engine).

Strategy:
  - Short leg: 30-delta put (same as V4)
  - Long leg: $10 below the short strike (defined risk)
  - Margin per spread = $10 x 100 = $1,000 (vs ~$2,000-$10,000 for CSP)
  - 7-day DTE (weekly rotation for faster premium capture)
  - 50% profit take (matches best_combo backtest — faster exits)
  - Max 30% of NAV in total spread margin
  - Max 2.5% of NAV per name, max 30 concurrent

Risk Controls (same as V4):
  - Bear gate: SPY < 50d SMA -> no new spreads
  - Equity curve brake: 60d lookback, 3% DD from peak -> scale to 25%
  - 2-day earnings buffer
  - VIX > 35 gate

Backtest results (2019-2026, BS pricing):
  - CAGR: ~154%, Sharpe: 4.09, Sortino: 4.65, MaxDD: -30.4%
  - Win rate: 63.6%, Profit factor: 2.07

Cost model:
  - $0.65/contract commission PER LEG ($1.30 per spread)
  - 2.5% slippage on premium (min $0.03/share)

Author: Claude (2026-07-05, post higher-returns study)
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

# HC #702 — Options pricing audit logger
try:
    from live_trading_linux.options_pricing_logger import log_option_price
    _HAS_PRICING_LOGGER = True
except ImportError:
    try:
        from options_pricing_logger import log_option_price
        _HAS_PRICING_LOGGER = True
    except ImportError:
        _HAS_PRICING_LOGGER = False

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "wheel_bps_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-BPS] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_bps_paper.log")),
    ],
)
log = logging.getLogger('WHEEL-BPS')

# ── Winning Configuration (from higher-returns study) ──
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Spread parameters
SPREAD_WIDTH = 10.0        # $10 between short and long strike
PUT_DELTA_TARGET = 0.30    # 30-delta short leg
DTE_TARGET = 7             # Weekly rotation (7 DTE)
DTE_MIN = 3
DTE_MAX = 12
PROFIT_TAKE_PCT = 0.65     # 65% profit take (raised from 50% — peak profit study: capture more of available premium)
BPS_STOP_LOSS_MULT = 1.0   # Close spread when loss reaches 1x premium collected
MIN_NET_CREDIT = 0.10      # Min $0.10/share net credit

# Position sizing (conservative — winning config used 30% margin cap)
MARGIN_CAP = 0.30          # Max 30% of NAV in total spread margin
PER_NAME_PCT = 0.025       # Max 2.5% of NAV per name (matches best_combo backtest)
MAX_CONCURRENT = 30        # Max 30 positions (matches best_combo backtest)

# Risk controls (same as V4)
VIX_MAX_GATE = 35.0
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
# Scales position size inversely with VIX instead of a hard binary gate.
VIX_SCALE_ENABLED = True   # Feature flag
VIX_SCALE_TIERS = [
    (15, 1.00),   # VIX < 15: full size (low vol = highest edge)
    (20, 0.80),   # VIX 15-20: 80% size
    (25, 0.50),   # VIX 20-25: half size (edge thinning)
    (30, 0.25),   # VIX 25-30: quarter size (edge near zero)
    (999, 0.00),  # VIX > 30: fully gated (negative Sharpe)
]

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65   # Per contract per leg
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# Price filter
MIN_PRICE = 10.0
MAX_PRICE = 500.0

# ── Sigma (volatility) filter ──
SIGMA_MAX_ENTRY = 0.8   # Skip tickers with 20d annualized vol > 80%

# ── Stop-loss ticker cooldown ──
STOP_LOSS_COOLDOWN_HOURS = 24

# ── Persistent loser blacklist ──
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

    # Fill-quality blacklist: tickers with consistently negative real credits
    # from 688 live option chain quotes (2026-07-09 analysis).
    FILL_QUALITY_BLACKLIST = {
        "IONS", "AEP", "HAL", "HON", "AMGN", "GD", "EXC", "ED",
        "CHWY", "ESS", "CL", "CARR", "EOG", "JD", "ET", "DOCU", "GOLD",
    }
    # Consistent backtest losers (from 400-trade Kelly analysis)
    # Updated 2026-07-15: added CRSP, AAL, CELH from live paper analysis
    BACKTEST_LOSERS = {"ABBV", "CROX", "CRSP", "AAL", "CELH"}
    tickers -= FILL_QUALITY_BLACKLIST
    tickers -= BACKTEST_LOSERS

    log.info(f"Universe loaded: {len(tickers)} tickers (fill-quality + loser blacklist applied)")
    return sorted(tickers)


# ── Earnings Filter ──
def load_earnings_dates():
    """Load cached earnings dates for the universe."""
    cache_path = ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet"
    if not cache_path.exists():
        log.warning("No cached earnings dates found")
        return {}

    df = pd.read_parquet(cache_path)
    lookup = {}
    date_col = "earnings_date" if "earnings_date" in df.columns else "date"
    for ticker, grp in df.groupby("ticker"):
        dates = pd.to_datetime(grp[date_col]).values
        dates = np.sort(dates)
        lookup[ticker] = dates

    log.info(f"Earnings dates loaded for {len(lookup)} tickers")
    return lookup


def has_earnings_soon(ticker, earnings_lookup, check_date=None, dte=DTE_TARGET, buffer=2):
    """Check if ticker has earnings within [today-buffer, today+DTE+buffer]."""
    if ticker not in earnings_lookup:
        return False

    if check_date is None:
        check_date = pd.Timestamp.now()
    else:
        check_date = pd.Timestamp(check_date)

    ed = earnings_lookup[ticker]
    window_start = np.datetime64(check_date) - np.timedelta64(buffer, 'D')
    window_end = np.datetime64(check_date) + np.timedelta64(dte + buffer, 'D')

    idx_s = np.searchsorted(ed, window_start, side='left')
    idx_e = np.searchsorted(ed, window_end, side='right')

    return idx_e > idx_s


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
    """
    Return position size scalar based on current VIX level.
    Research (2026-07-09): BPS edge is strong in low vol, disappears in high vol.
    Scaling instead of hard gate preserves moderate-vol edge at reduced size.
    """
    if not VIX_SCALE_ENABLED:
        return 1.0
    for threshold, scale in VIX_SCALE_TIERS:
        if vix < threshold:
            return scale
    return 0.0


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
                     earnings_lookup, universe, dd_trigger_active=False):
    """Open new BPS positions on tickers without existing positions."""
    if bear_mode:
        log.info("BEAR REGIME -- no new spreads")
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

    # Cross-engine dedup: avoid tickers already held by conservative BPS engine
    try:
        import json as _json
        _peer_state_path = Path(__file__).parent / 'wheel_bps_conservative_state' / 'state.json'
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

    opened = 0
    for ticker in candidates:
        # Check position cap at TOP of loop (fixes batch-exceeds-cap bug)
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
                    del state['stop_loss_cooldown'][ticker]
            except Exception:
                pass

        # Earnings filter (2-day buffer)
        if has_earnings_soon(ticker, earnings_lookup):
            continue

        # HC #750: Multi-signal confluence check (min 2 confirming signals)
        try:
            from live_trading_linux.signal_confluence import check_confluence, log_confluence
            _earnings_dt = None
            if ticker in earnings_lookup:
                _earnings_dt = earnings_lookup[ticker].strftime('%Y-%m-%d') if hasattr(earnings_lookup[ticker], 'strftime') else str(earnings_lookup[ticker])
            confluence = check_confluence(
                ticker, direction='short_put', price=S, vix=vix,
                sigma=sigma, earnings_date=_earnings_dt
            )
            if not confluence['pass']:
                log.debug(f"CONFLUENCE SKIP: {ticker} — {confluence['score']}/{confluence['min_required']} signals "
                          f"({', '.join(confluence['confirming_signals'])})")
                continue
            log.info(f"CONFLUENCE PASS: {ticker} — {confluence['score']} signals: "
                     f"{', '.join(confluence['confirming_signals'])}")
        except ImportError:
            pass  # Module not available, proceed without confluence check
        except Exception as _e:
            log.debug(f"Confluence check error for {ticker}: {_e}")

        # Find expiry
        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365.0

        # Find short strike (30-delta put)
        K_short = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')

        # Long strike = short strike - spread width
        K_long = K_short - SPREAD_WIDTH

        if K_long <= 0 or K_short <= 0:
            continue

        # Price both legs — Alpaca real pricing when available, BS fallback
        _exp_d = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
        if _HAS_PRICING_BRIDGE:
            prem_short = _alpaca_get_premium(ticker, S, K_short, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
            prem_long = _alpaca_get_premium(ticker, S, K_long, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
        else:
            prem_short = bs_price(S, K_short, T, sigma, kind='put')
            prem_long = bs_price(S, K_long, T, sigma, kind='put')
        net_prem_per_share = prem_short - prem_long

        if net_prem_per_share < MIN_NET_CREDIT:
            continue

        # Margin per contract = spread width x 100
        margin_per_contract = SPREAD_WIDTH * 100

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

        # Compute net credit after slippage and commissions (2 legs)
        prem_short_after_slip = apply_slippage(prem_short)
        prem_long_after_slip = prem_long * (1 + SLIPPAGE_FRAC)  # long leg costs more

        net_credit_per_share = prem_short_after_slip - prem_long_after_slip
        if net_credit_per_share <= 0:
            continue

        net_credit = net_credit_per_share * 100 * n_contracts
        # Commission: $0.65/contract per leg, 2 legs
        open_commission = 2 * COST_PER_CONTRACT * n_contracts
        net_credit -= open_commission

        if net_credit <= 0:
            continue

        # Execute: add net credit to cash (margin is held implicitly)
        state['cash'] += net_credit

        dte = (expiry_date - datetime.utcnow()).days
        state['spreads'].append({
            'ticker': ticker,
            'short_strike': K_short,
            'long_strike': K_long,
            'contracts': n_contracts,
            'premium_received': net_credit,  # total $ received after costs
            'sigma': sigma,
            'expiry': expiry_date.isoformat(),
            'open_date': datetime.utcnow().isoformat(),
            'spread_width': SPREAD_WIDTH,
            'margin_held': total_margin,
        })

        available_margin -= total_margin
        opened += 1

        log.info(f"SELL BPS {ticker} {K_short:.2f}/{K_long:.2f} "
                 f"({dte}d, {n_contracts}c), "
                 f"credit ${net_credit:.2f}, "
                 f"yield {net_credit/total_margin*100:.1f}%")
        log_trade({
            'action': 'sell_bps', 'ticker': ticker,
            'short_strike': K_short, 'long_strike': K_long,
            'contracts': n_contracts, 'net_credit': round(net_credit, 2),
            'dte': dte, 'sigma': round(sigma, 3),
            'underlying_price': round(S, 2),
            'spread_width': SPREAD_WIDTH,
            'brake_scale': brake_scale,
            'time': datetime.utcnow().isoformat(),
        })

        # HC #702 — log option pricing for both legs of the spread
        if _HAS_PRICING_LOGGER:
            log_option_price(
                engine="wheel-bps", ticker=ticker, spot=S, strike=K_short,
                expiry=expiry_date.isoformat(), option_type="put", iv_used=sigma,
                bs_price=prem_short, source="bs", action="entry_short_leg"
            )
            log_option_price(
                engine="wheel-bps", ticker=ticker, spot=S, strike=K_long,
                expiry=expiry_date.isoformat(), option_type="put", iv_used=sigma,
                bs_price=prem_long, source="bs", action="entry_long_leg"
            )

        if len(state['spreads']) >= MAX_CONCURRENT:
            break

    if opened:
        log.info(f"Opened {opened} new BPS (brake scale: {brake_scale:.0%}, "
                 f"total spreads: {len(state['spreads'])})")


# ── Main Loop ──
def run_cycle(universe, earnings_lookup):
    """Run one full check cycle."""
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

    # Bear mode: close all spreads early (buy back)
    if bear_mode:
        # BUG FIX: Collect spreads to close first, then remove after iteration.
        # Previously used state['spreads'].remove(sp) inside the loop, which
        # caused list mutation during iteration leading to duplicate closes
        # (34 duplicate trades / +$779 excess phantom PnL observed).
        bear_closed = []
        for sp in state['spreads']:
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

            # BUG FIX: Sanity check for inverted pricing. In a bull put spread,
            # short_strike > long_strike, so the short put must be worth MORE than
            # the long put. If short_val < long_val, the pricing bridge returned
            # inverted values — close_cost would go negative, generating phantom
            # profit (e.g., +$17,978 on a $10-wide spread with $2,000 max profit).
            if short_val < long_val:
                log.warning(f"PRICING INVERSION on {ticker}: short_val={short_val:.4f} < "
                            f"long_val={long_val:.4f} (strikes {sp['short_strike']}/{sp['long_strike']}). "
                            f"Skipping bear_close — pricing error.")
                continue

            close_cost = (short_val - long_val) * 100 * contracts
            close_fees = 2 * COST_PER_CONTRACT * contracts

            # BUG FIX: Cap close P&L at max possible profit for the spread.
            # Max profit on a bull put spread = net premium received.
            # Max P&L from closing = spread_width * 100 * contracts (entire spread value).
            spread_width = sp.get('spread_width', SPREAD_WIDTH)
            max_profit_cap = spread_width * 100 * contracts
            pnl = sp['premium_received'] - close_cost - close_fees
            if pnl > max_profit_cap:
                log.warning(f"PnL ${pnl:.2f} exceeds max profit cap ${max_profit_cap:.2f} on {ticker}. "
                            f"Clamping to cap. (close_cost={close_cost:.2f}, premium={sp['premium_received']:.2f})")
                pnl = max_profit_cap

            state['cash'] -= close_cost + close_fees
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            bear_closed.append(sp)

            log.info(f"BEAR CLOSE BPS {ticker}. PnL: ${pnl:.2f}")
            log_trade({
                'action': 'bear_close', 'ticker': ticker,
                'short_strike': sp['short_strike'], 'long_strike': sp['long_strike'],
                'pnl': round(pnl, 2), 'contracts': contracts,
                'time': pd.Timestamp.now().isoformat(),
            })

        # Remove closed spreads after iteration is complete
        for sp in bear_closed:
            state['spreads'].remove(sp)

    # Process existing spreads
    process_spreads(state, prices, sigmas)

    # Open new spreads
    open_new_spreads(state, prices, sigmas, vix, bear_mode, brake_scale,
                     earnings_lookup, universe, dd_trigger_active=dd_trigger_active)

    # Final NAV
    nav = compute_nav(state, prices, sigmas)
    check_nav_drop(nav)
    log_equity(state, nav)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    dd_status = "HALTED" if dd_trigger_active else "normal"
    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"spreads={len(state['spreads'])}, trades={state['trade_count']}, "
             f"dd_trigger={dd_status}")


def main():
    log.info("=" * 60)
    log.info("BULL PUT SPREAD (BPS) PAPER ENGINE")
    log.info(f"  Spread: ${SPREAD_WIDTH:.0f} wide, Delta: {PUT_DELTA_TARGET}")
    log.info(f"  DTE: {DTE_TARGET}d (weekly), PT: {PROFIT_TAKE_PCT:.0%}")
    log.info(f"  Margin: {MARGIN_CAP:.0%} cap, {PER_NAME_PCT:.0%} per name")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg, {SLIPPAGE_FRAC:.1%} slippage")
    log.info(f"  Expected: ~154% CAGR, Sharpe 4.09 (backtest)")
    log.info("=" * 60)

    # Initialize Alpaca real pricing (falls back to BS when unavailable)
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='wheel-bps')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    universe = load_universe()
    earnings_lookup = load_earnings_dates()

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
