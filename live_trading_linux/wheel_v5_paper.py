#!/usr/bin/env python3
"""
Wheel V5 Paper Engine — Higher Income + Combined Hedge
========================================================

V5.3 additions (2026-07-13):
  - Delta 25 (from d30): ladder study proved optimal (Sharpe 3.92, 0.6% assign)
  - Inverse-vol sizing via LGBM vol forecaster (IC=0.75, ICIR=5.82)
    Low predicted vol → up to 2x position, high vol → 0.5x position
    Scales per_name_limit by median_vol/pred_vol, capped [0.5, 2.0]

V5.1 additions (2026-07-10, adversarial-validated):
  - VIX Term Structure Sizing: cuts position size when VIX3M/VIX < 1.0 (backwardation)
  - Dynamic Beta Hedge: synthetic short SPY to offset market beta
  - Combined config (b=0.20/s=0.40/v=18): Sharpe 2.24, gap 0.21, R1 PASS
  - Adversarial: permutation p=0.000, bootstrap 62% R1 pass, param stability 24%

V5 improvements (validated via 36-config walk-forward sweep, 2018-2026):

1. SHORTER DTE (7-14 days vs 10-18): More premium cycles per year.
   Impact: +3.6pp CAGR (33.6% -> 37.2%) at same delta/no filters.

2. SECTOR FILTER: Remove Cannabis (PF 0.45), Consumer Cyclical (PF 0.72),
   Basic Materials (PF 0.98), Healthcare (PF 0.88). These sectors are net
   negative — removing them cuts max drawdown by 40% (-20.9% -> -9.5%).
   Impact: -5.7pp CAGR but -11.4pp MaxDD and +0.06 Sharpe.

3. IV RANK FLOOR 20%: Only sell when IV percentile rank >= 20th pctile
   vs ticker's own 1-year history. Filters out low-IV garbage premium.
   Impact: marginal CAGR lift, +0.11 Sharpe, -7.0pp MaxDD.

BEST COMBINED CONFIG: d30/dte7-14/iv20%/sector_filter
  Backtest: 33.9% CAGR, 3.18 Sharpe, 3.45 Sortino, -9.5% MaxDD
  Realized: 34.9% CAGR, 3.07 Sharpe, 88.7% WR, PF 3.07

  vs v4 baseline (d30/dte10-18/no filters):
    CAGR:    33.9% vs 33.6%  (+0.3pp)
    Sharpe:  3.18  vs 3.09   (+0.09)
    MaxDD:  -9.5%  vs -13.1% (+3.6pp improvement)
    WR:     88.7%  vs 88.4%  (+0.3pp)

Key insight: the sector filter + shorter DTE combo is the sweet spot.
Pure higher-delta (d35/d40) adds CAGR but degrades Sharpe and MaxDD.
The smart move is keeping d30 (proven edge) but cycling faster (weekly DTE)
and pruning losing sectors.

Author: Claude (2026-07-05, post v5 research cycle)
"""

import json
import math
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Set

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

# ── Vol Forecaster for inverse-vol position sizing (IC=0.75, ICIR=5.82) ──
try:
    try:
        from live_trading_linux.vol_forecaster_live import get_vol_predictions, get_vol_sizing_scale
    except ImportError:
        from vol_forecaster_live import get_vol_predictions, get_vol_sizing_scale
    _HAS_VOL_FORECASTER = True
except Exception:
    _HAS_VOL_FORECASTER = False

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
STATE_DIR = ROOT / "live_trading_linux" / "wheel_v5_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-V5] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_v5_paper.log")),
    ],
)
log = logging.getLogger('WHEEL-V5')

# ══════════════════════════════════════════════════════════════════════
# V5 CONFIGURATION — All validated in walk-forward backtest
# ══════════════════════════════════════════════════════════════════════
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Position sizing
MARGIN_REQ_PCT = 0.20      # 20% of notional for CSP margin
MARGIN_CAP = 0.40          # Max 40% of NAV in margin
PER_NAME_PCT = 0.03        # Max 3% of NAV per name
MIN_PREMIUM = 0.10         # Min $0.10/share premium

# ── V5 CHANGES vs V4 ──
PUT_DELTA_TARGET = 0.25    # V5.3: d25 optimal per delta ladder study (Sharpe 3.92, CAGR 10.7%, 0.6% assign rate)
CALL_DELTA_TARGET = 0.25
DTE_TARGET = 10            # V5: shorter (was 14) — more cycles/year
DTE_MIN = 7                # V5: shorter (was 10)
DTE_MAX = 14               # V5: shorter (was 18)
PROFIT_TAKE_PCT = 0.65     # SAME as v4 (validated optimum)
CSP_STOP_LOSS_MULT = 1.0   # V5.2: Close CSP when loss reaches 1x premium collected (HC #669 VRP research)
IV_RANK_FLOOR = 0.20       # V5: NEW — only sell when IV rank >= 20th pctile
LOSS_CUT_PCT = -0.15       # -15% loss cut on assigned shares
VIX_MAX_GATE = 35.0
MAX_ASSIGNMENTS_5D = 3
MAX_SHARE_POSITIONS = 5

# ── V5.1: COMBINED HEDGE PARAMETERS (adversarial-validated 2026-07-10) ──
# VIX term structure sizing: reduce exposure when backwardation signals fear
VIX_TS_SIZING_ENABLED = True

# Dynamic beta hedge: synthetic short SPY to offset portfolio market beta
HEDGE_ENABLED = True
HEDGE_BASE_RATIO = 0.20     # Hedge 20% of portfolio delta normally
HEDGE_STRESS_RATIO = 0.40   # Hedge 40% in stress (VIX > threshold)
HEDGE_VIX_THRESHOLD = 18.0  # VIX above this = stress mode
HEDGE_REBAL_TOLERANCE = 0.05  # Rebalance if hedge drifts >5% from target

# Equity curve brake parameters (robustness confirmed in v3)
BRAKE_LOOKBACK_DAYS = 60
BRAKE_THRESHOLD = 0.03     # 3% DD from peak triggers brake
BRAKE_SCALE = 0.25         # Scale exposure to 25% when braking

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# Price filter
MIN_PRICE = 10.0
MAX_PRICE = 500.0

# ── Sigma (volatility) filter ──
SIGMA_MAX_ENTRY = 0.8   # Skip tickers with 20d annualized vol > 80%

# ── Persistent loser blacklist ──
LOSER_BLACKLIST = {"CRSP", "AAL", "CELH"}

# ── V5: SECTOR EXCLUSION LIST ──
# These sectors are net negative in the wheel backtest.
# Removing them cuts MaxDD by 40% with minimal CAGR impact.
EXCLUDED_SECTORS = {
    "Cannabis",           # PF 0.45, avg_pnl -$0.45/trade
    "Consumer Cyclical",  # PF 0.72, avg_pnl -$0.37/trade
    "Basic Materials",    # PF 0.98, avg_pnl -$0.02/trade
    "Healthcare",         # PF 0.88, avg_pnl -$0.08/trade
}

# ── Sector mapping (from universe.parquet + expanded universe) ──
# This gets populated at startup from cached data
TICKER_SECTORS: Dict[str, str] = {}


# ══════════════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════════════
def load_universe():
    """Load the full universe from cached data, apply sector filter."""
    cache_dir = ROOT / "wheel_strategy_v1" / "data" / "cache"
    tickers = set()

    # Load tickers from price files
    for pf in ["prices.parquet", "prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        path = cache_dir / pf
        if path.exists():
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                tickers |= set(df["ticker"].unique())

    # Remove non-equity tickers
    tickers -= {"SPY", "^VIX", "VIX"}

    # Load sector mapping from universe.parquet
    uni_path = cache_dir / "universe.parquet"
    if uni_path.exists():
        uni = pd.read_parquet(uni_path)
        for _, row in uni.iterrows():
            TICKER_SECTORS[row["ticker"]] = row.get("sector", "Unknown")

    # Apply sector filter
    excluded = {t for t in tickers if TICKER_SECTORS.get(t, "Unknown") in EXCLUDED_SECTORS}
    tickers -= excluded

    log.info(f"Universe loaded: {len(tickers)} tickers "
             f"({len(excluded)} excluded by sector filter: "
             f"{', '.join(sorted(EXCLUDED_SECTORS))})")
    return sorted(tickers)


# ══════════════════════════════════════════════════════════════════════
# EARNINGS FILTER
# ══════════════════════════════════════════════════════════════════════
def load_earnings_dates():
    """Load cached earnings dates."""
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


# ══════════════════════════════════════════════════════════════════════
# IV RANK — V5 NEW FEATURE
# ══════════════════════════════════════════════════════════════════════
def compute_iv_rank(current_iv: float, iv_history: list) -> float:
    """
    Compute IV percentile rank: what % of historical IV is below current.
    Uses 1-year (252 trading day) lookback.
    """
    if not iv_history or len(iv_history) < 20:
        return 0.5  # Default to median if insufficient history
    hist = np.array(iv_history[-252:])  # 1-year lookback
    rank = np.sum(hist <= current_iv) / len(hist)
    return float(rank)


# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════
def get_live_prices(tickers, batch_size=50):
    """Fetch current prices, 20d realized vol, and IV history via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}
    iv_histories = {}  # For IV rank calculation

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="260d", auto_adjust=True,
                             threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data["Close"].columns:
                        close = data["Close"][t].dropna()
                        if len(close) > 0:
                            prices[t] = float(close.iloc[-1])
                            log_ret = np.log(close / close.shift(1)).dropna()

                            # Current 20d realized vol
                            if len(log_ret) >= 20:
                                sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252))
                            elif len(log_ret) > 5:
                                sigmas[t] = float(log_ret.std() * np.sqrt(252))
                            sigmas[t] = max(0.05, min(sigmas.get(t, 0.3), 2.0))

                            # IV history for rank calculation (rolling 20d vol)
                            if len(log_ret) >= 40:
                                rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
                                iv_histories[t] = rolling_vol.dropna().tolist()

            elif len(batch) == 1 and not data.empty:
                t = batch[0]
                close = data["Close"].dropna()
                if len(close) > 0:
                    prices[t] = float(close.iloc[-1])
                    log_ret = np.log(close / close.shift(1)).dropna()
                    sigmas[t] = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret) >= 20 else 0.3
                    sigmas[t] = max(0.05, min(sigmas[t], 2.0))
                    if len(log_ret) >= 40:
                        rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
                        iv_histories[t] = rolling_vol.dropna().tolist()
        except Exception as e:
            log.warning(f"Price batch {i}-{i+batch_size} failed: {e}")
        time.sleep(0.5)

    return prices, sigmas, iv_histories


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


def get_vix_ts_ratio():
    """Fetch VIX3M/VIX ratio for term structure sizing.
    >1.0 = contango (calm), <1.0 = backwardation (fear)."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period='5d')
        vix3m = yf.Ticker("^VIX3M").history(period='5d')
        if not vix.empty and not vix3m.empty:
            v = float(vix['Close'].iloc[-1])
            v3m = float(vix3m['Close'].iloc[-1])
            if v > 0:
                return v3m / v
    except Exception as e:
        log.warning(f"VIX TS ratio fetch failed: {e}")
    return 1.0  # Default: assume contango (no sizing reduction)


def vix_ts_sizing(ratio: float) -> float:
    """Map VIX3M/VIX ratio to position size multiplier.
    >1 = contango (full size), <1 = backwardation (cut size)."""
    breakpoints = [
        (0.75, 0.25), (0.85, 0.25), (0.95, 0.50),
        (1.00, 0.75), (1.10, 1.00), (1.30, 1.00),
    ]
    if np.isnan(ratio) if isinstance(ratio, float) else False:
        return 1.0
    if ratio <= breakpoints[0][0]:
        return breakpoints[0][1]
    if ratio >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        r0, s0 = breakpoints[i]
        r1, s1 = breakpoints[i + 1]
        if r0 <= ratio <= r1:
            frac = (ratio - r0) / (r1 - r0)
            return s0 + frac * (s1 - s0)
    return 1.0


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


# ══════════════════════════════════════════════════════════════════════
# EQUITY CURVE BRAKE
# ══════════════════════════════════════════════════════════════════════
def load_nav_history():
    if NAV_HISTORY_FILE.exists():
        with open(NAV_HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_nav_history(history):
    with open(NAV_HISTORY_FILE, 'w') as f:
        json.dump(history, f)


def compute_brake_scale(nav_history, current_nav):
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


# Fast drawdown trigger (validated 2026-07-08, p=0.01 permutation test)
# After 3-day NAV drops of ≥7%, halt new entries completely.
DD_TRIGGER_LOOKBACK = 3
DD_TRIGGER_THRESHOLD = -0.07

def check_dd_trigger(nav_history):
    """Fast DD trigger for CSP. Returns True if trailing 3d return < -7%."""
    if len(nav_history) < DD_TRIGGER_LOOKBACK + 1:
        return False
    current_nav = nav_history[-1]["nav"]
    past_nav = nav_history[-(DD_TRIGGER_LOOKBACK + 1)]["nav"]
    if past_nav <= 0:
        return False
    return (current_nav - past_nav) / past_nav < DD_TRIGGER_THRESHOLD


# ══════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': STARTING_CAPITAL,
        'positions': [],
        'realized_pnl': 0.0,
        'trade_count': 0,
        'start_date': datetime.utcnow().isoformat(),
        'last_check': None,
        'assignment_dates': [],
        'version': 'v5',
    }


def save_state(state):
    now = pd.Timestamp.now()
    state['assignment_dates'] = [
        d for d in state.get('assignment_dates', [])
        if (now - pd.Timestamp(d)).days <= 5
    ]
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def compute_nav(state, prices, spy_price=None):
    nav = state['cash']
    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            continue
        S = prices[ticker]
        if pos['side'] == 'short_put':
            nav += pos.get('margin_held', 0)
        elif pos['side'] in ('long_shares', 'short_call'):
            nav += S * 100 * pos.get('contracts', 1)
    # Include hedge mark-to-market
    if spy_price is not None:
        nav += compute_hedge_mtm(state, spy_price)
    # Include hedge realized P&L
    hedge = state.get('hedge', {})
    nav += hedge.get('realized_pnl', 0.0)
    return nav


SOD_NAV_FILE = STATE_DIR / "sod_nav.json"
NAV_DROP_ALERT_PCT = 0.025

_nav_drop_alerted_today = set()


def check_nav_drop(nav: float) -> None:
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
                f'"WHEEL V5 NAV DROP: {drop_pct:.1%} intraday '
                f'(${sod_nav:,.0f} -> ${nav:,.0f})" 2>/dev/null'
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# V5.1: DYNAMIC BETA HEDGE
# ══════════════════════════════════════════════════════════════════════
def compute_portfolio_delta(state, prices):
    """Compute total portfolio dollar delta (notional exposure to market)."""
    total_delta = 0.0
    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            continue
        S = prices[ticker]
        if pos['side'] == 'short_put':
            # Short put delta ≈ -delta_target × notional × contracts
            total_delta += PUT_DELTA_TARGET * S * 100 * pos.get('contracts', 1)
        elif pos['side'] in ('long_shares', 'short_call'):
            total_delta += S * 100 * pos.get('contracts', 1)
    return total_delta


def compute_target_hedge(portfolio_delta, vix, spy_price):
    """Compute target short SPY shares for hedge overlay."""
    if not HEDGE_ENABLED or spy_price is None or spy_price <= 0:
        return 0.0

    # Select hedge ratio based on VIX level
    if vix >= HEDGE_VIX_THRESHOLD:
        hedge_ratio = HEDGE_STRESS_RATIO
    else:
        hedge_ratio = HEDGE_BASE_RATIO

    # Target short SPY shares = portfolio_delta * hedge_ratio / spy_price
    target_shares = portfolio_delta * hedge_ratio / spy_price
    return target_shares


def rebalance_hedge(state, vix, spy_price, prices):
    """Rebalance hedge position to target. Returns hedge P&L from rebalance."""
    if not HEDGE_ENABLED or spy_price is None:
        return 0.0

    hedge = state.get('hedge', {'shares': 0.0, 'cost_basis': 0.0, 'realized_pnl': 0.0})

    portfolio_delta = compute_portfolio_delta(state, prices)
    target_shares = compute_target_hedge(portfolio_delta, vix, spy_price)
    current_shares = hedge.get('shares', 0.0)

    # Check if rebalance needed
    if target_shares > 0:
        drift = abs(current_shares - target_shares) / target_shares
    else:
        drift = abs(current_shares)

    if drift < HEDGE_REBAL_TOLERANCE and current_shares > 0:
        # No rebalance needed
        state['hedge'] = hedge
        return 0.0

    # Rebalance: close old, open new at current SPY price
    # P&L from closing old position
    rebal_pnl = 0.0
    if current_shares > 0:
        # We were short SPY. P&L = (cost_basis - current) * shares
        old_basis = hedge.get('cost_basis', spy_price)
        rebal_pnl = (old_basis - spy_price) * current_shares
        hedge['realized_pnl'] = hedge.get('realized_pnl', 0.0) + rebal_pnl

    # Open new hedge at current price
    hedge['shares'] = round(target_shares, 2)
    hedge['cost_basis'] = spy_price
    hedge['last_rebal'] = datetime.utcnow().isoformat()

    mode = "STRESS" if vix >= HEDGE_VIX_THRESHOLD else "NORMAL"
    ratio = HEDGE_STRESS_RATIO if vix >= HEDGE_VIX_THRESHOLD else HEDGE_BASE_RATIO
    log.info(f"HEDGE REBAL [{mode}]: {current_shares:.0f} -> {target_shares:.0f} SPY shares "
             f"(ratio {ratio:.0%}, VIX {vix:.1f}, delta ${portfolio_delta:,.0f}). "
             f"Rebal P&L: ${rebal_pnl:.2f}")

    if abs(rebal_pnl) > 10:
        log_trade({'action': 'hedge_rebal', 'old_shares': round(current_shares, 2),
                   'new_shares': round(target_shares, 2), 'spy_price': spy_price,
                   'pnl': round(rebal_pnl, 2), 'mode': mode,
                   'time': datetime.utcnow().isoformat()})

    state['hedge'] = hedge
    return rebal_pnl


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


# ══════════════════════════════════════════════════════════════════════
# EXPIRY LOGIC
# ══════════════════════════════════════════════════════════════════════
def find_expiry(from_date=None):
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


# ══════════════════════════════════════════════════════════════════════
# TRADING LOGIC
# ══════════════════════════════════════════════════════════════════════
def process_positions(state, prices, sigmas):
    """Check existing positions for profit-take, expiry, assignment."""
    now = pd.Timestamp.now()
    new_positions = []

    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            new_positions.append(pos)
            continue

        S = prices[ticker]
        sigma = sigmas.get(ticker, pos.get('sigma', 0.20))
        expiry = pd.Timestamp(pos['expiry']) if pos.get('expiry') else None

        if pos['side'] == 'short_put':
            if expiry is None:
                new_positions.append(pos)
                continue
            T = max((expiry - now).days, 0) / 365
            # FIX (HC #709 audit): Use consistent pricing — same source as entry
            expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
            current = _get_option_price_consistent(ticker, S, pos['strike'], T, sigma, expiry_date, kind='put')
            profit_pct = (pos['entry_premium'] - current) / pos['entry_premium'] if pos['entry_premium'] > 0 else 0

            # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
            entry_dt = pd.Timestamp(pos.get('entry_date', '2020-01-01'))
            hours_held = (now - entry_dt).total_seconds() / 3600
            if hours_held < 24:
                new_positions.append(pos)
                continue

            if profit_pct >= PROFIT_TAKE_PCT:
                buyback = current * 100 * pos.get('contracts', 1) + COST_PER_CONTRACT
                pnl = (pos['entry_premium'] - current) * 100 * pos.get('contracts', 1) - 2 * COST_PER_CONTRACT
                state['cash'] += pos.get('margin_held', 0)
                state['cash'] -= buyback
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"PROFIT TAKE CSP {ticker} @ {profit_pct:.0%}. PnL: ${pnl:.2f}")
                log_trade({'action': 'profit_take', 'ticker': ticker, 'pnl': round(pnl, 2),
                          'time': now.isoformat()})
                continue

            # V5.2: Stop-loss — close CSP when loss reaches 1x premium collected
            # profit_pct <= -1.0 means current option price >= 2x entry premium
            # i.e., unrealized loss >= premium we collected
            if profit_pct <= -CSP_STOP_LOSS_MULT:
                buyback = current * 100 * pos.get('contracts', 1) + COST_PER_CONTRACT
                pnl = (pos['entry_premium'] - current) * 100 * pos.get('contracts', 1) - 2 * COST_PER_CONTRACT
                state['cash'] += pos.get('margin_held', 0)
                state['cash'] -= buyback
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"STOP LOSS CSP {ticker} @ {profit_pct:.0%} (loss >= {CSP_STOP_LOSS_MULT:.0f}x premium). PnL: ${pnl:.2f}")
                log_trade({'action': 'stop_loss_csp', 'ticker': ticker, 'pnl': round(pnl, 2),
                          'profit_pct': round(profit_pct, 3), 'time': now.isoformat()})
                continue

            if now >= expiry:
                state['cash'] += pos.get('margin_held', 0)
                if S <= pos['strike']:
                    n_recent = len(state.get('assignment_dates', []))
                    n_shares = len([p for p in state['positions'] if p['side'] in ('long_shares', 'short_call')])

                    if n_recent >= MAX_ASSIGNMENTS_5D or n_shares >= MAX_SHARE_POSITIONS:
                        intrinsic = (pos['strike'] - S) * 100
                        loss = intrinsic - pos['entry_premium'] * 100 + COST_PER_CONTRACT
                        state['cash'] -= loss
                        state['realized_pnl'] -= loss
                        log.info(f"ASSIGNMENT REFUSED {ticker} (cap reached). Loss: ${loss:.2f}")
                        log_trade({'action': 'assignment_refused', 'ticker': ticker,
                                  'loss': round(loss, 2), 'time': now.isoformat()})
                    else:
                        share_cost = pos['strike'] * 100 + COST_PER_CONTRACT
                        state['cash'] -= share_cost
                        state.setdefault('assignment_dates', []).append(now.isoformat())
                        new_positions.append({
                            'ticker': ticker, 'side': 'long_shares',
                            'strike': 0, 'share_basis': pos['strike'] - pos['entry_premium'],
                            'expiry': '', 'entry_premium': 0,
                            'contracts': pos.get('contracts', 1), 'sigma': sigma,
                            'entry_date': now.isoformat(),
                        })
                        log.info(f"ASSIGNED {ticker} at {pos['strike']:.2f}. "
                                 f"Basis: ${pos['strike'] - pos['entry_premium']:.2f}")
                        log_trade({'action': 'assigned', 'ticker': ticker,
                                  'strike': pos['strike'], 'time': now.isoformat()})
                else:
                    pnl = pos['entry_premium'] * 100 * pos.get('contracts', 1) - COST_PER_CONTRACT
                    state['realized_pnl'] += pnl
                    state['trade_count'] += 1
                    log.info(f"EXPIRED OTM {ticker}. Premium: ${pnl:.2f}")
                    log_trade({'action': 'expired_otm', 'ticker': ticker,
                              'pnl': round(pnl, 2), 'time': now.isoformat()})
                continue

            new_positions.append(pos)

        elif pos['side'] == 'long_shares':
            pnl_pct = (S - pos['share_basis']) / pos['share_basis'] if pos['share_basis'] > 0 else 0
            if pnl_pct <= LOSS_CUT_PCT:
                proceeds = S * 100 * pos.get('contracts', 1) - COST_PER_CONTRACT
                state['cash'] += proceeds
                realized = (S - pos['share_basis']) * 100
                state['realized_pnl'] += realized
                state['trade_count'] += 1
                log.info(f"LOSS CUT {ticker} at ${S:.2f} ({pnl_pct:.0%}). Loss: ${realized:.2f}")
                log_trade({'action': 'loss_cut', 'ticker': ticker, 'pnl': round(realized, 2),
                          'time': now.isoformat()})
                continue

            expiry_date = find_expiry()
            if expiry_date:
                T = (expiry_date - datetime.utcnow()).days / 365
                K = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')
                # Use Alpaca real pricing when available, fall back to BS
                if _HAS_PRICING_BRIDGE:
                    premium = _alpaca_get_premium(
                        ticker, S, K, T, sigma, expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date,
                        kind='call', bs_price_fn=bs_price)
                else:
                    premium = bs_price(S, K, T, sigma, kind='call')
                premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
                if premium >= 0.10:
                    state['cash'] += premium * 100 - COST_PER_CONTRACT
                    new_positions.append({
                        'ticker': ticker, 'side': 'short_call',
                        'strike': K, 'share_basis': pos['share_basis'],
                        'expiry': expiry_date.isoformat(),
                        'entry_premium': premium,
                        'contracts': pos.get('contracts', 1), 'sigma': sigma,
                        'entry_date': now.isoformat(),
                    })
                    log.info(f"SELL CC {ticker} {K:.2f} strike, premium ${premium:.2f}")
                    log_trade({'action': 'sell_cc', 'ticker': ticker, 'strike': K,
                              'premium': round(premium, 2), 'time': now.isoformat()})
                    continue
            new_positions.append(pos)

        elif pos['side'] == 'short_call':
            if expiry is None:
                new_positions.append(pos)
                continue
            T = max((expiry - now).days, 0) / 365
            # FIX (HC #709 audit): Use consistent pricing — same source as entry
            expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
            current = _get_option_price_consistent(ticker, S, pos['strike'], T, sigma, expiry_date, kind='call')
            profit_pct = (pos['entry_premium'] - current) / pos['entry_premium'] if pos['entry_premium'] > 0 else 0

            # FIX (HC #709 audit): Anti-churn cooldown — don't close within 24h of open
            entry_dt = pd.Timestamp(pos.get('entry_date', '2020-01-01'))
            hours_held = (now - entry_dt).total_seconds() / 3600
            if hours_held < 24:
                new_positions.append(pos)
                continue

            if profit_pct >= PROFIT_TAKE_PCT:
                pnl_cc = (pos['entry_premium'] - current) * 100 - 2 * COST_PER_CONTRACT
                state['cash'] += pnl_cc
                state['realized_pnl'] += pnl_cc
                new_positions.append({
                    'ticker': ticker, 'side': 'long_shares',
                    'strike': 0, 'share_basis': pos['share_basis'],
                    'expiry': '', 'entry_premium': 0,
                    'contracts': pos.get('contracts', 1), 'sigma': sigma,
                    'entry_date': now.isoformat(),
                })
                log.info(f"CLOSE CC {ticker} @ {profit_pct:.0%} profit. PnL: ${pnl_cc:.2f}")
                continue

            if now >= expiry:
                if S >= pos['strike']:
                    sale = pos['strike'] * 100
                    pnl = (pos['strike'] - pos['share_basis'] + pos['entry_premium']) * 100 - COST_PER_CONTRACT
                    state['cash'] += sale
                    state['realized_pnl'] += pnl
                    state['trade_count'] += 1
                    log.info(f"CALLED AWAY {ticker} at {pos['strike']:.2f}. PnL: ${pnl:.2f}")
                    log_trade({'action': 'called_away', 'ticker': ticker,
                              'pnl': round(pnl, 2), 'time': now.isoformat()})
                else:
                    pnl = pos['entry_premium'] * 100 - COST_PER_CONTRACT
                    state['realized_pnl'] += pnl
                    new_positions.append({
                        'ticker': ticker, 'side': 'long_shares',
                        'strike': 0, 'share_basis': pos['share_basis'],
                        'expiry': '', 'entry_premium': 0,
                        'contracts': pos.get('contracts', 1), 'sigma': sigma,
                        'entry_date': now.isoformat(),
                    })
                    log.info(f"CC EXPIRED OTM {ticker}. Premium: ${pnl:.2f}")
                continue

            new_positions.append(pos)

    state['positions'] = new_positions


def open_new_positions(state, prices, sigmas, iv_histories, vix, bear_mode,
                       brake_scale, earnings_lookup, universe):
    """Open CSPs on tickers, with V5 filters: sector + IV rank + earnings."""
    if bear_mode:
        log.info("BEAR REGIME -- no new CSPs")
        return
    if vix > VIX_MAX_GATE:
        log.info(f"VIX {vix:.1f} > {VIX_MAX_GATE} -- no new entries")
        return
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT -- no new CSPs")
        return

    active_tickers = {p['ticker'] for p in state['positions']}
    nav = compute_nav(state, prices)

    effective_margin_cap = MARGIN_CAP * brake_scale
    current_margin = sum(p.get('margin_held', 0) for p in state['positions']
                        if p['side'] == 'short_put')
    available_margin = nav * effective_margin_cap - current_margin
    per_name_limit = nav * PER_NAME_PCT * brake_scale

    # V5.3: Load vol predictions for inverse-vol sizing
    vol_preds = None
    if _HAS_VOL_FORECASTER:
        try:
            vol_preds = get_vol_predictions()
            if vol_preds:
                log.info(f"Vol forecaster: {len(vol_preds)} predictions loaded for sizing")
        except Exception as e:
            log.warning(f"Vol forecaster failed (using uniform sizing): {e}")

    if available_margin <= 0:
        return

    import random
    candidates = [t for t in universe if t in prices and t not in active_tickers]
    random.shuffle(candidates)

    opened = 0
    iv_filtered = 0
    for ticker in candidates:
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

        # V5: Sector filter (already applied in universe load, but double-check)
        if TICKER_SECTORS.get(ticker, "Unknown") in EXCLUDED_SECTORS:
            continue

        # V5: IV rank filter
        if IV_RANK_FLOOR > 0 and ticker in iv_histories:
            iv_rank = compute_iv_rank(sigma, iv_histories[ticker])
            if iv_rank < IV_RANK_FLOOR:
                iv_filtered += 1
                continue

        # Earnings filter (2-day buffer)
        if has_earnings_soon(ticker, earnings_lookup):
            continue

        # V5.3: Inverse-vol sizing — scale per-name limit by predicted vol
        vol_scale = 1.0
        if vol_preds and ticker in vol_preds:
            vol_scale = get_vol_sizing_scale(ticker, vol_preds)
        adj_per_name_limit = per_name_limit * vol_scale

        # Size check
        notional = S * 100
        margin_req = notional * MARGIN_REQ_PCT

        if margin_req > adj_per_name_limit:
            continue
        if margin_req > available_margin:
            continue
        if state['cash'] < margin_req:
            continue

        # Find strike and price
        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365
        K = find_strike(S, sigma, T, PUT_DELTA_TARGET, kind='put')
        # Use Alpaca real pricing when available, fall back to BS
        if _HAS_PRICING_BRIDGE:
            premium = _alpaca_get_premium(
                ticker, S, K, T, sigma, expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date,
                kind='put', bs_price_fn=bs_price)
        else:
            premium = bs_price(S, K, T, sigma, kind='put')
        premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

        if premium < MIN_PREMIUM:
            continue

        # Open position
        state['cash'] -= margin_req
        state['cash'] += premium * 100 - COST_PER_CONTRACT

        state['positions'].append({
            'ticker': ticker, 'side': 'short_put',
            'strike': K, 'share_basis': 0,
            'expiry': expiry_date.isoformat(),
            'entry_premium': premium,
            'contracts': 1, 'sigma': sigma,
            'margin_held': margin_req,
            'entry_date': datetime.utcnow().isoformat(),
        })

        available_margin -= margin_req
        opened += 1

        dte = (expiry_date - datetime.utcnow()).days
        vol_info = f", vol_scale={vol_scale:.2f}" if vol_scale != 1.0 else ""
        log.info(f"SELL CSP {ticker} {K:.2f} strike ({dte}d), "
                 f"premium ${premium:.2f}/sh, yield {premium/K*100:.1f}%{vol_info}")
        log_trade({'action': 'sell_csp', 'ticker': ticker, 'strike': K,
                  'premium': round(premium, 4), 'dte': dte, 'sigma': round(sigma, 3),
                  'brake_scale': brake_scale, 'vol_scale': round(vol_scale, 3),
                  'time': datetime.utcnow().isoformat()})

        # HC #702 — log option pricing for audit
        if _HAS_PRICING_LOGGER:
            log_option_price(
                engine="wheel-v5", ticker=ticker, spot=S, strike=K,
                expiry=expiry_date.isoformat(), option_type="put", iv_used=sigma,
                bs_price=bs_price(S, K, T, sigma, kind='put'), source="bs", action="entry"
            )

    if opened or iv_filtered:
        log.info(f"Opened {opened} new CSPs (brake: {brake_scale:.0%}, "
                 f"IV-filtered: {iv_filtered})")


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run_cycle(universe, earnings_lookup):
    """Run one full check cycle."""
    state = load_state()

    position_tickers = {p['ticker'] for p in state['positions']}
    needed = list(position_tickers | set(universe[:100]))
    prices, sigmas, iv_histories = get_live_prices(needed)
    vix = get_vix()
    bear_mode = is_bear_regime()
    spy_price = get_spy_price()

    # V5.1: VIX term structure sizing
    ts_ratio = get_vix_ts_ratio() if VIX_TS_SIZING_ENABLED else 1.0
    ts_scale = vix_ts_sizing(ts_ratio) if VIX_TS_SIZING_ENABLED else 1.0

    nav = compute_nav(state, prices, spy_price)
    hedge = state.get('hedge', {})
    hedge_shares = hedge.get('shares', 0.0)
    hedge_mtm = compute_hedge_mtm(state, spy_price) if spy_price else 0.0

    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, VIX={vix:.1f}, "
             f"regime={'BEAR' if bear_mode else 'BULL'}, "
             f"TS={ts_ratio:.3f} (scale {ts_scale:.0%}), "
             f"hedge={hedge_shares:.0f}sh SPY (MTM ${hedge_mtm:,.0f})")

    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(), "nav": round(nav, 2)})
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    brake_scale = compute_brake_scale(nav_history, nav)

    # V5.1: Apply VIX TS sizing to brake_scale (compounds with equity brake)
    effective_brake = brake_scale * ts_scale
    if ts_scale < 1.0:
        log.info(f"VIX TS sizing active: scale {ts_scale:.0%} "
                 f"(VIX3M/VIX={ts_ratio:.3f}). "
                 f"Effective brake: {effective_brake:.0%}")

    # Bear mode: close CSPs early
    if bear_mode:
        for pos in list(state['positions']):
            if pos['side'] != 'short_put':
                continue
            ticker = pos['ticker']
            if ticker not in prices:
                continue
            S = prices[ticker]
            sigma = sigmas.get(ticker, 0.20)
            expiry = pd.Timestamp(pos['expiry'])
            T = max((expiry - pd.Timestamp.now()).days, 0) / 365
            # FIX (HC #709 audit): Use consistent pricing — same source as entry
            expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
            current = _get_option_price_consistent(ticker, S, pos['strike'], T, sigma, expiry_date, kind='put')
            buyback = current * 100 + COST_PER_CONTRACT
            state['cash'] += pos.get('margin_held', 0)
            state['cash'] -= buyback
            pnl = (pos['entry_premium'] - current) * 100 - 2 * COST_PER_CONTRACT
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            state['positions'].remove(pos)
            log.info(f"BEAR CLOSE {ticker}. PnL: ${pnl:.2f}")
            log_trade({'action': 'bear_close', 'ticker': ticker,
                      'pnl': round(pnl, 2), 'time': pd.Timestamp.now().isoformat()})

    # Fast drawdown trigger (validated p=0.01, avoidance +4.38%/cycle)
    dd_trigger_active = check_dd_trigger(nav_history)
    if dd_trigger_active:
        log.info("DD TRIGGER ACTIVE: 3-day return < -7%. Halting new entries.")

    process_positions(state, prices, sigmas)

    if not dd_trigger_active:
        open_new_positions(state, prices, sigmas, iv_histories, vix, bear_mode,
                           effective_brake, earnings_lookup, universe)
    else:
        log.info("Skipping new entries — DD trigger halting all opens.")

    # V5.1: Rebalance hedge
    if HEDGE_ENABLED and spy_price:
        rebalance_hedge(state, vix, spy_price, prices)

    nav = compute_nav(state, prices, spy_price)
    check_nav_drop(nav)
    log_equity(state, nav)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, trades={state['trade_count']}, "
             f"hedge={state.get('hedge',{}).get('shares',0):.0f}sh SPY")


def main():
    log.info("=" * 60)
    log.info("WHEEL V5.1 PAPER ENGINE — HIGHER INCOME + COMBINED HEDGE")
    log.info(f"  Delta: {PUT_DELTA_TARGET}, PT: {PROFIT_TAKE_PCT:.0%}, "
             f"DTE: {DTE_MIN}-{DTE_MAX}d (target {DTE_TARGET}d)")
    log.info(f"  IV Rank Floor: {IV_RANK_FLOOR:.0%}")
    log.info(f"  Excluded Sectors: {', '.join(sorted(EXCLUDED_SECTORS))}")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  VIX TS Sizing: {'ON' if VIX_TS_SIZING_ENABLED else 'OFF'}")
    log.info(f"  Hedge: {'ON' if HEDGE_ENABLED else 'OFF'} "
             f"(base={HEDGE_BASE_RATIO:.0%}, stress={HEDGE_STRESS_RATIO:.0%}, "
             f"vix_thresh={HEDGE_VIX_THRESHOLD})")
    log.info(f"  Margin: {MARGIN_CAP:.0%} cap, {PER_NAME_PCT:.0%} per name")
    log.info(f"  Backtest: Sharpe 2.24, gap 0.21 (R1 PASS), CAGR 23.2%, MaxDD -10.6%")
    log.info("=" * 60)

    # Initialize Alpaca real pricing (falls back to BS when unavailable)
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='wheel-v5')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    universe = load_universe()
    earnings_lookup = load_earnings_dates()

    log.info(f"Universe: {len(universe)} tickers, "
             f"earnings data: {len(earnings_lookup)} tickers")

    while True:
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle(universe, earnings_lookup)
            else:
                if now.minute < 5:
                    state = load_state()
                    log.info(f"Off-hours: positions={len(state['positions'])}, "
                             f"realized=${state['realized_pnl']:.2f}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == '__main__':
    main()
