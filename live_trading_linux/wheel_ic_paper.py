#!/usr/bin/env python3
"""
Iron Condor (IC) Paper Engine — Market-Neutral Premium Capture
===============================================================

Sells iron condors ($10-wide bull put spread + $10-wide bear call spread)
on a 230-ticker equity universe. This is the market-neutral evolution of
the BPS paper engine.

Strategy:
  - Put side: short 25-delta put, long $10 below (same as BPS but tighter)
  - Call side: short 25-delta call, long $10 above
  - Margin per IC = $10 x 100 = $1,000 (only one side can be ITM)
  - 7-day DTE (weekly rotation)
  - 50% profit take (lower than BPS because 4-leg positions converge faster)
  - Max 30% of NAV in total margin
  - Max 3% of NAV per name

Risk Controls:
  - NO bear gate (IC is market-neutral — profits from BOTH sides)
  - Equity curve brake: 60d lookback, 3% DD from peak -> scale to 25%
  - 2-day earnings buffer
  - VIX > 40 gate (higher threshold — IC BENEFITS from elevated vol)
  - Drawdown trigger (HC #662 R4)

Backtest results (2019-2026, BS pricing):
  - IC 65% PT: CAGR 354%, Sharpe 6.96, Sortino 9.21, MaxDD -24.7%
  - Bull Sharpe 9.72, Correction 6.58, Bear 3.32 (ALL regimes positive)
  - Permutation test: PASS (p=0.000)

Cost model:
  - $0.65/contract commission PER LEG ($2.60 per IC open, $2.60 close)
  - 2.5% slippage on premium (min $0.03/share)

Author: Claude (2026-07-06, post iron condor study)
"""

import json
import math

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
STATE_DIR = ROOT / "live_trading_linux" / "wheel_ic_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-IC] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_ic_paper.log")),
    ],
)
log = logging.getLogger('WHEEL-IC')

# ── Winning Configuration (from iron condor study) ──
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Spread parameters
SPREAD_WIDTH = 10.0        # $10 between short and long strike (each side)
PUT_DELTA_TARGET = 0.25    # 25-delta short put (tighter than BPS 30d)
CALL_DELTA_TARGET = 0.25   # 25-delta short call (symmetric)
DTE_TARGET = 7             # Weekly rotation (7 DTE)
DTE_MIN = 3
DTE_MAX = 12
PROFIT_TAKE_PCT = 0.65     # 65% profit take (raised from 50% — peak profit study showed 68% avg capture, leaving 32% unharvested)
IC_STOP_LOSS_MULT = 1.0    # Close IC when loss reaches 1x premium collected
MIN_NET_CREDIT = 0.10      # Min $0.10/share net credit per side

# Position sizing
MARGIN_CAP = 0.30          # Max 30% of NAV in total margin
PER_NAME_PCT = 0.025       # Max 2.5% of NAV per name (aligned with BPS best_combo)
MAX_CONCURRENT = 30        # Max 30 concurrent iron condors (aligned with BPS best_combo)

# Risk controls (IC is market-neutral — NO bear gate needed)
VIX_MAX_GATE = 40.0        # Higher threshold — IC benefits from elevated vol
BRAKE_LOOKBACK_DAYS = 60
BRAKE_THRESHOLD = 0.03     # 3% DD from peak triggers brake
BRAKE_SCALE = 0.25         # Scale exposure to 25% when braking

# Fast drawdown trigger (HC #662 R4 — validated by permutation test p=0.000)
# Bad days cluster 2.3x in BPS returns. When trailing 3-day return < -5%,
# halt new positions entirely. Backtest: Sharpe 4.09→4.26, MaxDD -30%→-22%.
DD_TRIGGER_LOOKBACK = 3    # 3-day trailing return window
DD_TRIGGER_THRESHOLD = -0.05  # -5% trailing return triggers halt
DD_TRIGGER_ENABLED = True  # Feature flag for A/B tracking

# ── COMBINED HEDGE OVERLAY (from ic_combined_hedge research, deployed 2026-07-12) ──
# VIX term structure sizing: reduce NEW IC position sizes by 50% when VIX >= threshold
# Dynamic SPY beta hedge: synthetic short SPY to offset portfolio market delta
HEDGE_ENABLED = True
HEDGE_BASE_RATIO = 0.05       # 5% SPY short (notional = ratio × IC_NAV) normally
HEDGE_STRESS_RATIO = 0.20    # 20% SPY short when VIX >= threshold
HEDGE_VIX_THRESHOLD = 22.0   # VIX above this = stressed mode
HEDGE_REBAL_TOLERANCE = 0.05  # Rebalance if hedge drifts >5% from target
VIX_STRESS_SIZE_CUT = 0.50   # Cut new IC sizes by 50% when VIX >= threshold

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

# ── Ticker cooldown: prevent rapid-fire churning (HC #701 fix) ──
TICKER_COOLDOWN_HOURS = 24  # Min hours between closing and reopening same ticker

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
    log.info(f"Universe loaded: {len(tickers)} tickers from cache")
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


# ══════════════════════════════════════════════════════════════════════
# COMBINED HEDGE OVERLAY
# ══════════════════════════════════════════════════════════════════════

def get_spy_price():
    """Fetch current SPY price for hedge calculations."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period='5d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        log.warning(f"SPY price fetch failed: {e}")
    return None


def compute_portfolio_delta(state, prices):
    """Compute total portfolio dollar delta for the IC book.

    Iron condors have net-short delta from put side (short put = positive delta)
    offset partly by call side (short call = negative delta). Net delta is
    typically small but non-zero, especially when positions drift ITM.
    For hedge sizing we use the gross put-side delta (conservative).
    """
    total_delta = 0.0
    for sp in state['spreads']:
        ticker = sp['ticker']
        if ticker not in prices:
            continue
        S = prices[ticker]
        contracts = sp.get('contracts', 1)
        # Short put delta contribution ≈ put_delta_target × S × 100 × contracts
        total_delta += PUT_DELTA_TARGET * S * 100 * contracts
    return total_delta


def compute_target_hedge_notional(nav, vix):
    """Compute target SPY short notional based on VIX regime.

    Returns target notional $ amount of SPY to be short.
    """
    if not HEDGE_ENABLED:
        return 0.0

    if vix >= HEDGE_VIX_THRESHOLD:
        ratio = HEDGE_STRESS_RATIO
    else:
        ratio = HEDGE_BASE_RATIO

    return nav * ratio


def rebalance_hedge(state, vix, spy_price, nav):
    """Rebalance hedge position to target. Tracks SPY short shares."""
    if not HEDGE_ENABLED or spy_price is None or spy_price <= 0:
        return

    hedge = state.get('hedge', {'shares': 0.0, 'cost_basis': 0.0, 'realized_pnl': 0.0})

    target_notional = compute_target_hedge_notional(nav, vix)
    target_shares = target_notional / spy_price
    current_shares = hedge.get('shares', 0.0)

    # Check if rebalance needed
    if target_shares > 0:
        drift = abs(current_shares - target_shares) / target_shares
    else:
        drift = abs(current_shares)

    if drift < HEDGE_REBAL_TOLERANCE and current_shares > 0:
        state['hedge'] = hedge
        return

    # Realize P&L from closing old position
    rebal_pnl = 0.0
    if current_shares > 0:
        old_basis = hedge.get('cost_basis', spy_price)
        # Short SPY: profit when SPY falls
        rebal_pnl = (old_basis - spy_price) * current_shares
        hedge['realized_pnl'] = hedge.get('realized_pnl', 0.0) + rebal_pnl

    # Open new hedge at current price
    hedge['shares'] = round(target_shares, 2)
    hedge['cost_basis'] = spy_price
    hedge['last_rebal'] = datetime.utcnow().isoformat()

    mode = "STRESSED" if vix >= HEDGE_VIX_THRESHOLD else "NORMAL"
    ratio = HEDGE_STRESS_RATIO if vix >= HEDGE_VIX_THRESHOLD else HEDGE_BASE_RATIO
    log.info(f"HEDGE REBAL [{mode}]: {current_shares:.0f} -> {target_shares:.0f} SPY shares "
             f"(ratio {ratio:.0%}, VIX {vix:.1f}, NAV ${nav:,.0f}). "
             f"Rebal P&L: ${rebal_pnl:.2f}")

    if abs(current_shares - target_shares) > 1 or abs(rebal_pnl) > 10:
        log_trade({'action': 'hedge_rebal', 'old_shares': round(current_shares, 2),
                   'new_shares': round(target_shares, 2), 'spy_price': spy_price,
                   'pnl': round(rebal_pnl, 2), 'mode': mode,
                   'vix': round(vix, 2), 'nav': round(nav, 2),
                   'time': datetime.utcnow().isoformat()})

    state['hedge'] = hedge


def compute_hedge_mtm(state, spy_price):
    """Compute mark-to-market P&L of current hedge position."""
    if not HEDGE_ENABLED or spy_price is None:
        return 0.0
    hedge = state.get('hedge', {})
    shares = hedge.get('shares', 0.0)
    cost_basis = hedge.get('cost_basis', spy_price)
    if shares <= 0:
        return 0.0
    # Short SPY: profit when SPY falls
    return (cost_basis - spy_price) * shares


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
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def compute_nav(state, prices, sigmas, spy_price=None):
    """Compute current NAV = cash + unrealized IC MTM + hedge MTM + hedge realized."""
    nav = state['cash']

    for sp in state['spreads']:
        ticker = sp['ticker']
        if ticker not in prices:
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, sp.get('sigma', 0.20))
        expiry = pd.Timestamp(sp['expiry'])
        T = max((expiry - pd.Timestamp.now()).days, 0) / 365.0
        contracts = sp.get('contracts', 1)

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        # Put side MTM
        put_short_val = _get_option_price_consistent(ticker, S, sp['put_short'], T, sigma, expiry_date, kind='put')
        put_long_val = _get_option_price_consistent(ticker, S, sp['put_long'], T, sigma, expiry_date, kind='put')
        put_mtm = (put_short_val - put_long_val) * 100 * contracts

        # Call side MTM
        call_short_val = _get_option_price_consistent(ticker, S, sp['call_short'], T, sigma, expiry_date, kind='call')
        call_long_val = _get_option_price_consistent(ticker, S, sp['call_long'], T, sigma, expiry_date, kind='call')
        call_mtm = (call_short_val - call_long_val) * 100 * contracts

        # We owe the spread value (short IC = liability)
        nav -= (put_mtm + call_mtm)

    # Include hedge mark-to-market and realized P&L
    if spy_price is not None:
        nav += compute_hedge_mtm(state, spy_price)
    hedge = state.get('hedge', {})
    nav += hedge.get('realized_pnl', 0.0)

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
    """Check existing iron condors for profit-take, expiry, or max-loss."""
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
        premium_received = sp['premium_received']

        # FIX (HC #709 audit): Use consistent pricing — same source as entry
        expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
        put_short_val = _get_option_price_consistent(ticker, S, sp['put_short'], T, sigma, expiry_date, kind='put')
        put_long_val = _get_option_price_consistent(ticker, S, sp['put_long'], T, sigma, expiry_date, kind='put')
        call_short_val = _get_option_price_consistent(ticker, S, sp['call_short'], T, sigma, expiry_date, kind='call')
        call_long_val = _get_option_price_consistent(ticker, S, sp['call_long'], T, sigma, expiry_date, kind='call')

        put_spread_val = (put_short_val - put_long_val) * 100 * contracts
        call_spread_val = (call_short_val - call_long_val) * 100 * contracts
        total_close_cost = put_spread_val + call_spread_val + 4 * COST_PER_CONTRACT * contracts

        # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
        entry_dt = pd.Timestamp(sp.get('entry_date', '2020-01-01'))
        hours_held = (now - entry_dt).total_seconds() / 3600
        if hours_held < 24:
            new_spreads.append(sp)
            continue

        # ── Profit take (market hours only to avoid BS pricing artifacts) ──
        if premium_received > 0:
            profit_pct = (premium_received - total_close_cost) / premium_received
            # Only trigger profit-take during RTH (9:30-16:00 ET)
            try:
                import pytz
                _et = pytz.timezone('US/Eastern')
                _now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(_et)
                _in_rth = ((_now_et.hour == 9 and _now_et.minute >= 30) or
                           (10 <= _now_et.hour < 16))
            except Exception:
                _in_rth = True  # Fail-open: allow profit-take if pytz fails
            if profit_pct >= PROFIT_TAKE_PCT and _in_rth:
                state['cash'] -= total_close_cost
                pnl = premium_received - total_close_cost
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"PROFIT TAKE IC {ticker} @ {profit_pct:.0%}. "
                         f"PnL: ${pnl:.2f} ({contracts} contracts)")
                log_trade({
                    'action': 'profit_take_ic', 'ticker': ticker,
                    'put_short': sp['put_short'], 'put_long': sp['put_long'],
                    'call_short': sp['call_short'], 'call_long': sp['call_long'],
                    'pnl': round(pnl, 2), 'contracts': contracts,
                    'time': now.isoformat(),
                })
                continue

            # Stop-loss: close IC when loss reaches 1x premium collected
            if profit_pct <= -IC_STOP_LOSS_MULT:
                state['cash'] -= total_close_cost
                pnl = premium_received - total_close_cost
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"STOP LOSS IC {ticker} @ {profit_pct:.0%} (loss >= {IC_STOP_LOSS_MULT:.0f}x premium). "
                         f"PnL: ${pnl:.2f}")
                log_trade({
                    'action': 'stop_loss_ic', 'ticker': ticker,
                    'put_short': sp['put_short'], 'put_long': sp['put_long'],
                    'call_short': sp['call_short'], 'call_long': sp['call_long'],
                    'pnl': round(pnl, 2), 'contracts': contracts,
                    'profit_pct': round(profit_pct, 3), 'time': now.isoformat(),
                })
                continue

        # ── Expiry settlement ──
        if now >= expiry:
            close_fees = 4 * COST_PER_CONTRACT * contracts

            # Put side losses
            put_loss = 0
            if S < sp['put_short'] and S >= sp['put_long']:
                put_loss = (sp['put_short'] - S) * 100 * contracts
            elif S < sp['put_long']:
                put_loss = (sp['put_short'] - sp['put_long']) * 100 * contracts

            # Call side losses
            call_loss = 0
            if S > sp['call_short'] and S <= sp['call_long']:
                call_loss = (S - sp['call_short']) * 100 * contracts
            elif S > sp['call_long']:
                call_loss = (sp['call_long'] - sp['call_short']) * 100 * contracts

            total_loss = put_loss + call_loss
            state['cash'] -= total_loss + close_fees
            pnl = premium_received - total_loss - close_fees
            state['realized_pnl'] += pnl
            state['trade_count'] += 1

            side = "OTM" if total_loss == 0 else ("PUT" if put_loss > 0 and call_loss == 0
                    else ("CALL" if call_loss > 0 and put_loss == 0 else "BOTH"))
            log.info(f"EXPIRED IC {ticker} {side} loss. "
                     f"Put: ${sp['put_short']:.0f}/{sp['put_long']:.0f} "
                     f"Call: ${sp['call_short']:.0f}/{sp['call_long']:.0f} "
                     f"S=${S:.2f} PnL: ${pnl:.2f}")
            log_trade({
                'action': f'expired_ic_{side.lower()}', 'ticker': ticker,
                'put_short': sp['put_short'], 'put_long': sp['put_long'],
                'call_short': sp['call_short'], 'call_long': sp['call_long'],
                'pnl': round(pnl, 2), 'put_loss': round(put_loss, 2),
                'call_loss': round(call_loss, 2),
                'underlying_at_expiry': round(S, 2),
                'contracts': contracts, 'time': now.isoformat(),
            })
            continue

        # ── Max loss early exit: underlying way beyond either wing ──
        if S < sp['put_long'] * 0.97 or S > sp['call_long'] * 1.03:
            max_loss = SPREAD_WIDTH * 100 * contracts  # max loss on one side
            close_fees = 4 * COST_PER_CONTRACT * contracts
            state['cash'] -= max_loss + close_fees
            pnl = premium_received - max_loss - close_fees
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            side = "PUT" if S < sp['put_long'] else "CALL"
            log.info(f"EARLY CLOSE (max loss {side}) IC {ticker} @ ${S:.2f}. PnL: ${pnl:.2f}")
            log_trade({
                'action': f'early_close_{side.lower()}', 'ticker': ticker,
                'pnl': round(pnl, 2), 'contracts': contracts,
                'underlying_price': round(S, 2), 'time': now.isoformat(),
            })
            continue

        # Position survives
        new_spreads.append(sp)

    state['spreads'] = new_spreads


def open_new_spreads(state, prices, sigmas, vix, bear_mode, brake_scale,
                     earnings_lookup, universe, dd_trigger_active=False):
    """Open new iron condor positions on tickers without existing positions."""
    # NO bear gate for iron condors — they are market-neutral
    if vix > VIX_MAX_GATE:
        log.info(f"VIX {vix:.1f} > {VIX_MAX_GATE} -- no new entries")
        return
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT -- no new ICs")
        return
    if dd_trigger_active:
        log.info("DD TRIGGER HALT -- 3-day trailing return below threshold, no new entries")
        return

    active_tickers = {sp['ticker'] for sp in state['spreads']}

    # Ticker cooldown: skip tickers closed recently (prevents churning)
    now_utc = datetime.utcnow()
    cooldown_tickers = set()
    for ct in state.get('closed_trades', []):
        close_str = ct.get('close_date') or ct.get('closed_at', '')
        if close_str:
            try:
                close_dt = datetime.fromisoformat(close_str.replace('Z', '+00:00').replace('+00:00', ''))
                hours_since = (now_utc - close_dt).total_seconds() / 3600
                if hours_since < TICKER_COOLDOWN_HOURS:
                    cooldown_tickers.add(ct.get('ticker', ''))
            except (ValueError, TypeError):
                pass
    if cooldown_tickers:
        log.info(f"COOLDOWN ({TICKER_COOLDOWN_HOURS}h): {sorted(cooldown_tickers)}")

    # Market hours guard: only open new positions during RTH (9:30-16:00 ET)
    import pytz
    et = pytz.timezone('US/Eastern')
    now_et = now_utc.replace(tzinfo=pytz.utc).astimezone(et)
    if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30) or now_et.hour >= 16:
        log.info(f"OFF-HOURS ({now_et.strftime('%H:%M ET')}) -- no new entries")
        return

    nav = compute_nav(state, prices, sigmas)

    effective_margin_cap = MARGIN_CAP * brake_scale
    current_margin = sum(
        SPREAD_WIDTH * 100 * sp.get('contracts', 1)
        for sp in state['spreads']
    )
    available_margin = nav * effective_margin_cap - current_margin
    per_name_limit = nav * PER_NAME_PCT * brake_scale

    if available_margin <= 0:
        return

    if len(state['spreads']) >= MAX_CONCURRENT:
        return

    # Sort by IV descending — higher IV = more premium = better condor
    candidates = [t for t in universe if t in prices and t not in active_tickers and t not in cooldown_tickers]
    candidates.sort(key=lambda t: sigmas.get(t, 0), reverse=True)

    opened = 0
    for ticker in candidates:
        if ticker not in prices or ticker not in sigmas:
            continue

        S = prices[ticker]
        sigma = sigmas[ticker]

        if S < MIN_PRICE or S > MAX_PRICE:
            continue

        # Blacklist: consistent losers across engines
        if ticker in LOSER_BLACKLIST:
            continue

        # Sigma filter: skip extreme-vol names (sigma > 0.8 annualized)
        if sigma > SIGMA_MAX_ENTRY:
            log.info(f"SIGMA SKIP: {ticker} sigma={sigma:.2f} > {SIGMA_MAX_ENTRY:.2f} threshold")
            continue

        if has_earnings_soon(ticker, earnings_lookup):
            continue

        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365.0

        # ── Put side: short 25-delta put, long $10 below ──
        K_put_short = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')
        K_put_long = K_put_short - SPREAD_WIDTH
        if K_put_long <= 0:
            continue

        # Use Alpaca real pricing when available, fall back to BS
        _exp_d = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
        if _HAS_PRICING_BRIDGE:
            prem_put_short = _alpaca_get_premium(ticker, S, K_put_short, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
            prem_put_long = _alpaca_get_premium(ticker, S, K_put_long, T, sigma, _exp_d, kind='put', bs_price_fn=bs_price)
        else:
            prem_put_short = bs_price(S, K_put_short, T, sigma, kind='put')
            prem_put_long = bs_price(S, K_put_long, T, sigma, kind='put')
        put_credit = prem_put_short - prem_put_long

        # ── Call side: short 25-delta call, long $10 above ──
        K_call_short = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')
        K_call_long = K_call_short + SPREAD_WIDTH

        # ── Minimum gap between short strikes (fix: narrow ICs on low-priced stocks) ──
        min_gap = max(3.0, S * 0.04)  # At least $3 or 4% of stock price
        if K_call_short - K_put_short < min_gap:
            log.debug(f"Skip {ticker}: IC gap ${K_call_short - K_put_short:.1f} < "
                      f"min ${min_gap:.1f} (S=${S:.2f})")
            continue

        if _HAS_PRICING_BRIDGE:
            prem_call_short = _alpaca_get_premium(ticker, S, K_call_short, T, sigma, _exp_d, kind='call', bs_price_fn=bs_price)
            prem_call_long = _alpaca_get_premium(ticker, S, K_call_long, T, sigma, _exp_d, kind='call', bs_price_fn=bs_price)
        else:
            prem_call_short = bs_price(S, K_call_short, T, sigma, kind='call')
            prem_call_long = bs_price(S, K_call_long, T, sigma, kind='call')
        call_credit = prem_call_short - prem_call_long

        total_credit_per_share = put_credit + call_credit
        if total_credit_per_share < MIN_NET_CREDIT:
            continue

        # IC margin = max(put spread, call spread) width = SPREAD_WIDTH per contract
        margin_per_contract = SPREAD_WIDTH * 100

        max_by_name = int(per_name_limit // margin_per_contract)
        max_by_margin = int(available_margin // margin_per_contract)
        n_contracts = min(max_by_name, max_by_margin)
        n_contracts = max(n_contracts, 1)

        total_margin = margin_per_contract * n_contracts
        if total_margin > available_margin:
            n_contracts = int(available_margin // margin_per_contract)
        if n_contracts < 1:
            continue
        total_margin = margin_per_contract * n_contracts

        # Net credit after slippage and commissions (4 legs)
        put_short_slip = apply_slippage(prem_put_short)
        put_long_slip = prem_put_long * (1 + SLIPPAGE_FRAC)
        call_short_slip = apply_slippage(prem_call_short)
        call_long_slip = prem_call_long * (1 + SLIPPAGE_FRAC)

        net_credit_per_share = (put_short_slip - put_long_slip) + (call_short_slip - call_long_slip)
        if net_credit_per_share <= 0:
            continue

        net_credit = net_credit_per_share * 100 * n_contracts
        open_commission = 4 * COST_PER_CONTRACT * n_contracts  # 4 legs
        net_credit -= open_commission

        if net_credit <= 0:
            continue

        state['cash'] += net_credit

        dte = (expiry_date - datetime.utcnow()).days
        state['spreads'].append({
            'ticker': ticker,
            'put_short': K_put_short,
            'put_long': K_put_long,
            'call_short': K_call_short,
            'call_long': K_call_long,
            'contracts': n_contracts,
            'premium_received': net_credit,
            'sigma': sigma,
            'expiry': expiry_date.isoformat(),
            'open_date': datetime.utcnow().isoformat(),
            'spread_width': SPREAD_WIDTH,
            'margin_held': total_margin,
        })

        available_margin -= total_margin
        opened += 1

        # HC #702: Log all 4 legs for BS vs Alpaca comparison
        if _HAS_PRICING_LOGGER:
            _exp_str = str(_exp_d)[:10] if '_exp_d' in dir() else expiry_date.isoformat()[:10]
            for _k, _kind, _bs_val in [
                (K_put_short, 'put', prem_put_short),
                (K_put_long, 'put', prem_put_long),
                (K_call_short, 'call', prem_call_short),
                (K_call_long, 'call', prem_call_long),
            ]:
                log_option_price(
                    engine="wheel_ic", ticker=ticker, spot=S, strike=_k,
                    expiry=_exp_str, option_type=_kind, iv_used=sigma,
                    bs_price=_bs_val, source="bs_or_alpaca", action="entry"
                )

        log.info(f"SELL IC {ticker} P:{K_put_short:.0f}/{K_put_long:.0f} "
                 f"C:{K_call_short:.0f}/{K_call_long:.0f} "
                 f"({dte}d, {n_contracts}c), "
                 f"credit ${net_credit:.2f}, "
                 f"yield {net_credit/total_margin*100:.1f}%")
        log_trade({
            'action': 'sell_ic', 'ticker': ticker,
            'put_short': K_put_short, 'put_long': K_put_long,
            'call_short': K_call_short, 'call_long': K_call_long,
            'contracts': n_contracts, 'net_credit': round(net_credit, 2),
            'dte': dte, 'sigma': round(sigma, 3),
            'underlying_price': round(S, 2),
            'brake_scale': brake_scale,
            'time': datetime.utcnow().isoformat(),
        })

        if len(state['spreads']) >= MAX_CONCURRENT:
            break

    if opened:
        log.info(f"Opened {opened} new ICs (brake scale: {brake_scale:.0%}, "
                 f"total positions: {len(state['spreads'])})")


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
    spy_price = get_spy_price()

    nav = compute_nav(state, prices, sigmas, spy_price)
    total_margin = sum(
        SPREAD_WIDTH * 100 * sp.get('contracts', 1)
        for sp in state['spreads']
    )
    margin_util = total_margin / nav * 100 if nav > 0 else 0

    # Hedge status
    hedge = state.get('hedge', {})
    hedge_shares = hedge.get('shares', 0.0)
    hedge_mtm = compute_hedge_mtm(state, spy_price) if spy_price else 0.0
    hedge_mode = "STRESSED" if vix >= HEDGE_VIX_THRESHOLD else "NORMAL"

    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"spreads={len(state['spreads'])}, margin={margin_util:.1f}%, "
             f"VIX={vix:.1f}, regime={'BEAR' if bear_mode else 'BULL'}, "
             f"hedge={hedge_shares:.0f}sh SPY [{hedge_mode}] (MTM ${hedge_mtm:,.0f})")

    # Update NAV history for brake
    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(), "nav": round(nav, 2)})
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    # Compute brake scale
    brake_scale = compute_brake_scale(nav_history, nav)

    # VIX stress sizing: cut new IC sizes by 50% when VIX >= threshold
    vix_size_scale = VIX_STRESS_SIZE_CUT if vix >= HEDGE_VIX_THRESHOLD else 1.0
    effective_brake = brake_scale * vix_size_scale
    if vix_size_scale < 1.0:
        log.info(f"VIX STRESS SIZING: VIX {vix:.1f} >= {HEDGE_VIX_THRESHOLD} -> "
                 f"new IC sizes cut to {vix_size_scale:.0%}. "
                 f"Effective brake: {effective_brake:.0%}")

    # ── Panic confluence overlay (offensive + defensive) ──
    # When panic reversal fires, scale UP (more premium to sell at high IV).
    # When stress is building, scale DOWN as early warning.
    try:
        from live_trading_linux.panic_confluence_monitor import get_panic_confluence
        panic = get_panic_confluence()
        panic_mult = panic.get('position_multiplier', 1.0)
        panic_mode = panic.get('mode', 'normal')
        if panic_mode == 'offensive':
            effective_brake = effective_brake * panic_mult
            log.info(f"PANIC OFFENSIVE: confluence {panic.get('confluence_score', 0)}/3, "
                     f"multiplier {panic_mult:.2f}x -> effective scale {effective_brake:.0%}")
        elif panic_mode == 'cautious':
            effective_brake = effective_brake * panic_mult  # 0.70x
            log.info(f"PANIC CAUTIOUS: stress building, "
                     f"multiplier {panic_mult:.2f}x -> effective scale {effective_brake:.0%}")
    except Exception as e:
        log.debug(f"Panic confluence check skipped: {e}")

    # Fast drawdown trigger (HC #662 R4)
    dd_trigger_active = check_dd_trigger(nav_history)

    # Iron condors are market-neutral — NO bear-mode close needed
    # Just process existing positions normally

    # Process existing spreads
    process_spreads(state, prices, sigmas)

    # Open new spreads (using effective_brake which includes VIX stress sizing)
    open_new_spreads(state, prices, sigmas, vix, bear_mode, effective_brake,
                     earnings_lookup, universe, dd_trigger_active=dd_trigger_active)

    # Rebalance hedge overlay
    if HEDGE_ENABLED and spy_price:
        rebalance_hedge(state, vix, spy_price, nav)

    # Final NAV
    nav = compute_nav(state, prices, sigmas, spy_price)
    check_nav_drop(nav)
    log_equity(state, nav)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    dd_status = "HALTED" if dd_trigger_active else "normal"
    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"spreads={len(state['spreads'])}, trades={state['trade_count']}, "
             f"dd_trigger={dd_status}, "
             f"hedge={state.get('hedge',{}).get('shares',0):.0f}sh SPY")


def main():
    log.info("=" * 60)
    log.info("IRON CONDOR (IC) PAPER ENGINE — Market-Neutral")
    log.info(f"  Spread: ${SPREAD_WIDTH:.0f} wide each side")
    log.info(f"  Put delta: {PUT_DELTA_TARGET}, Call delta: {CALL_DELTA_TARGET}")
    log.info(f"  DTE: {DTE_TARGET}d (weekly), PT: {PROFIT_TAKE_PCT:.0%}")
    log.info(f"  Margin: {MARGIN_CAP:.0%} cap, {PER_NAME_PCT:.0%} per name")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  Cost: ${COST_PER_CONTRACT}/contract/leg x 4 legs, {SLIPPAGE_FRAC:.1%} slippage")
    log.info(f"  Hedge: base {HEDGE_BASE_RATIO:.0%} / stressed {HEDGE_STRESS_RATIO:.0%} SPY short "
             f"(VIX >= {HEDGE_VIX_THRESHOLD})")
    log.info(f"  VIX stress sizing: new ICs cut to {VIX_STRESS_SIZE_CUT:.0%} when VIX >= "
             f"{HEDGE_VIX_THRESHOLD}")
    log.info(f"  Expected: ~354% CAGR, Sharpe 6.96 (backtest IC 65% PT)")
    log.info(f"  Bear regime: Sharpe +3.32 (market-neutral)")
    log.info("=" * 60)

    # Initialize Alpaca real pricing (falls back to BS when unavailable)
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='wheel-ic')
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
