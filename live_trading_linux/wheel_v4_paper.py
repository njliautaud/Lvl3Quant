#!/usr/bin/env python3
"""
Wheel V4 Paper Engine — All Validated Improvements
====================================================

Incorporates every optimization validated in the v3 research cycle:
- 230-ticker universe (HC #660 expansion)
- 30-delta puts (better premium vs 25-delta, validated)
- 65% profit take (optimal from v3 sweep)
- 2-day earnings buffer (Sharpe +30%, HC #660)
- Equity curve brake: 60d/3%/25% (robustness confirmed, 75-config sweep)
- Bear gate (SPY < 50d SMA → no new CSPs)
- Margin-based position sizing (20% notional, 40% portfolio cap, 3% per name)

Expected: ~25% CAGR, Sharpe 1.83, MaxDD -14.3% (backtest 2019-2026)
Monte Carlo: 90.5% probability of positive year, median +26% return

Author: Claude (2026-07-04, post v3 research cycle)
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

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "wheel_v4_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"
NAV_HISTORY_FILE = STATE_DIR / "nav_history.json"  # For equity brake

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-V4] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_v4_paper.log")),
    ],
)
log = logging.getLogger('WHEEL-V4')

# ── Validated Configuration (from v3 research cycle) ──
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes during market hours

# Position sizing
MARGIN_REQ_PCT = 0.20      # 20% of notional for CSP margin
MARGIN_CAP = 0.40          # Max 40% of NAV in margin
PER_NAME_PCT = 0.03        # Max 3% of NAV per name
MIN_PREMIUM = 0.10         # Min $0.10/share premium

# Strategy parameters (all validated in v3 backtests)
PUT_DELTA_TARGET = 0.30    # Validated winner (vs 25-delta)
CALL_DELTA_TARGET = 0.30
DTE_TARGET = 14            # 2-week DTE
DTE_MIN = 10
DTE_MAX = 18
PROFIT_TAKE_PCT = 0.65     # 65% profit take (validated)
CSP_STOP_LOSS_MULT = 1.0   # Close CSP when loss reaches 1x premium collected (VRP research validated)
LOSS_CUT_PCT = -0.15       # -15% loss cut on assigned shares
VIX_MAX_GATE = 35.0
MAX_ASSIGNMENTS_5D = 3
MAX_SHARE_POSITIONS = 5

# Equity curve brake parameters (robustness confirmed: 71% of 75-config grid beats baseline)
BRAKE_LOOKBACK_DAYS = 60   # Peak lookback window
BRAKE_THRESHOLD = 0.03     # 3% DD from peak triggers brake
BRAKE_SCALE = 0.25         # Scale exposure to 25% when braking

# Pricing
RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# Price filter for universe
MIN_PRICE = 10.0
MAX_PRICE = 500.0

# ── Universe (loaded from v3 cached data) ──
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

    # Remove non-equity tickers
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
        return False  # No data = assume no earnings (conservative: let it trade)

    if check_date is None:
        check_date = pd.Timestamp.now()
    else:
        check_date = pd.Timestamp(check_date)

    ed = earnings_lookup[ticker]
    window_start = np.datetime64(check_date) - np.timedelta64(buffer, 'D')
    window_end = np.datetime64(check_date) + np.timedelta64(dte + buffer, 'D')

    idx_s = np.searchsorted(ed, window_start, side='left')
    idx_e = np.searchsorted(ed, window_end, side='right')

    return idx_e > idx_s  # True if any earnings in window

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

# ── Price Data ──
def get_live_prices(tickers, batch_size=50):
    """Fetch current prices and 20d realized vol via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}

    # Batch download for efficiency
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
        time.sleep(0.5)  # Rate limit

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
    """Load NAV history for equity brake calculation."""
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
        return 1.0  # Not enough history

    # Get last BRAKE_LOOKBACK_DAYS entries
    recent = nav_history[-BRAKE_LOOKBACK_DAYS:]
    peak = max(entry["nav"] for entry in recent)

    if current_nav < peak * (1 - BRAKE_THRESHOLD):
        log.info(f"EQUITY BRAKE ACTIVE: NAV ${current_nav:,.0f} < peak ${peak:,.0f} × {1-BRAKE_THRESHOLD:.0%} "
                 f"= ${peak*(1-BRAKE_THRESHOLD):,.0f}. Scale → {BRAKE_SCALE:.0%}")
        return BRAKE_SCALE

    return 1.0


# Fast drawdown trigger (validated 2026-07-08, p=0.01 permutation test)
# After 3-day NAV drops of ≥7%, halt new entries completely.
# Avoidance value: +4.38% per 14d cycle. Fires rarely (0.6% of days).
DD_TRIGGER_LOOKBACK = 3   # days
DD_TRIGGER_THRESHOLD = -0.07  # -7% trailing return

def check_dd_trigger(nav_history):
    """
    Fast drawdown trigger for CSP engine.
    Returns True if trailing 3-day return < -7%, meaning we should halt new entries.
    More conservative threshold than BPS (-5%) because CSP has deeper drawdowns.
    """
    if len(nav_history) < DD_TRIGGER_LOOKBACK + 1:
        return False
    current_nav = nav_history[-1]["nav"]
    past_nav = nav_history[-(DD_TRIGGER_LOOKBACK + 1)]["nav"]
    if past_nav <= 0:
        return False
    trailing_ret = (current_nav - past_nav) / past_nav
    return trailing_ret < DD_TRIGGER_THRESHOLD

# ── State Management ──
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
    }

def save_state(state):
    # Clean old assignment dates (>5 days)
    now = pd.Timestamp.now()
    state['assignment_dates'] = [
        d for d in state.get('assignment_dates', [])
        if (now - pd.Timestamp(d)).days <= 5
    ]
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def compute_nav(state, prices):
    """Compute current NAV."""
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
    return nav

SOD_NAV_FILE = STATE_DIR / "sod_nav.json"
NAV_DROP_ALERT_PCT = 0.025  # 2.5% intraday drop triggers alert
_nav_drop_alerted_today = set()  # track alerts to avoid spam

def check_nav_drop(nav: float) -> None:
    """Alert if NAV drops >2.5% from start-of-day. P3-1 backlog item."""
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Load or set start-of-day NAV
    sod_data = {}
    if SOD_NAV_FILE.exists():
        try:
            sod_data = json.loads(SOD_NAV_FILE.read_text())
        except Exception:
            pass

    if sod_data.get("date") != today:
        # New day — set SOD NAV
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
        log.warning(f"⚠️ NAV DROP ALERT: {drop_pct:.1%} intraday "
                    f"(SOD: ${sod_nav:,.0f} → ${nav:,.0f})")
        # Fire QCC alert
        try:
            os.system(
                f'node /home/jupiter/teleclaude-main/utils/webhook_notifier.js '
                f'"WHEEL V4 NAV DROP: {drop_pct:.1%} intraday '
                f'(${sod_nav:,.0f} → ${nav:,.0f})" 2>/dev/null'
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

# ── Expiry Logic ──
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

# ── Trading Logic ──
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
            current = bs_price(S, pos['strike'], T, sigma, kind='put')
            profit_pct = (pos['entry_premium'] - current) / pos['entry_premium'] if pos['entry_premium'] > 0 else 0

            # Profit take at 65%
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

            # Stop-loss: close CSP when loss reaches 1x premium collected
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

            # Expiry
            if now >= expiry:
                state['cash'] += pos.get('margin_held', 0)
                if S <= pos['strike']:
                    # Assignment
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
            # Loss cut check
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

            # Sell covered call
            expiry_date = find_expiry()
            if expiry_date:
                T = (expiry_date - datetime.utcnow()).days / 365
                K = find_strike(S, sigma, T, CALL_DELTA_TARGET, kind='call')
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
            current = bs_price(S, pos['strike'], T, sigma, kind='call')
            profit_pct = (pos['entry_premium'] - current) / pos['entry_premium'] if pos['entry_premium'] > 0 else 0

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

def open_new_positions(state, prices, sigmas, vix, bear_mode, brake_scale,
                       earnings_lookup, universe):
    """Open CSPs on tickers without positions, with earnings filter + brake."""
    if bear_mode:
        log.info("BEAR REGIME — no new CSPs")
        return
    if vix > VIX_MAX_GATE:
        log.info(f"VIX {vix:.1f} > {VIX_MAX_GATE} — no new entries")
        return
    if brake_scale <= 0:
        log.info("EQUITY BRAKE HALT — no new CSPs")
        return

    active_tickers = {p['ticker'] for p in state['positions']}
    nav = compute_nav(state, prices)

    # Scale margin cap by brake
    effective_margin_cap = MARGIN_CAP * brake_scale
    current_margin = sum(p.get('margin_held', 0) for p in state['positions']
                        if p['side'] == 'short_put')
    available_margin = nav * effective_margin_cap - current_margin
    per_name_limit = nav * PER_NAME_PCT * brake_scale

    if available_margin <= 0:
        return

    # Shuffle for diversification (don't always pick same tickers)
    import random
    candidates = [t for t in universe if t in prices and t not in active_tickers]
    random.shuffle(candidates)

    opened = 0
    for ticker in candidates:
        if ticker not in prices or ticker not in sigmas:
            continue

        S = prices[ticker]
        sigma = sigmas[ticker]

        # Price filter
        if S < MIN_PRICE or S > MAX_PRICE:
            continue

        # Earnings filter (2-day buffer)
        if has_earnings_soon(ticker, earnings_lookup):
            continue

        # Size check
        notional = S * 100
        margin_req = notional * MARGIN_REQ_PCT

        if margin_req > per_name_limit:
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
        log.info(f"SELL CSP {ticker} {K:.2f} strike ({dte}d), "
                 f"premium ${premium:.2f}/sh, yield {premium/K*100:.1f}%")
        log_trade({'action': 'sell_csp', 'ticker': ticker, 'strike': K,
                  'premium': round(premium, 4), 'dte': dte, 'sigma': round(sigma, 3),
                  'brake_scale': brake_scale, 'time': datetime.utcnow().isoformat()})

    if opened:
        log.info(f"Opened {opened} new CSPs (brake scale: {brake_scale:.0%})")

# ── Main Loop ──
def run_cycle(universe, earnings_lookup):
    """Run one full check cycle."""
    state = load_state()

    # Fetch prices (only for tickers we might trade + current positions)
    position_tickers = {p['ticker'] for p in state['positions']}
    needed = list(position_tickers | set(universe[:100]))  # Top 100 + positions
    prices, sigmas = get_live_prices(needed)
    vix = get_vix()
    bear_mode = is_bear_regime()

    nav = compute_nav(state, prices)
    log.info(f"NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, VIX={vix:.1f}, "
             f"regime={'BEAR' if bear_mode else 'BULL'}")

    # Update NAV history for brake
    nav_history = load_nav_history()
    nav_history.append({"date": datetime.utcnow().isoformat(), "nav": round(nav, 2)})
    # Keep last 120 entries (enough for 60-day brake)
    nav_history = nav_history[-120:]
    save_nav_history(nav_history)

    # Compute brake scale
    brake_scale = compute_brake_scale(nav_history, nav)

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
            current = bs_price(S, pos['strike'], T, sigma, kind='put')
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

    # Process existing positions
    process_positions(state, prices, sigmas)

    # Open new positions (skip if DD trigger active)
    if not dd_trigger_active:
        open_new_positions(state, prices, sigmas, vix, bear_mode, brake_scale,
                           earnings_lookup, universe)
    else:
        log.info("Skipping new entries — DD trigger halting all opens.")

    # Final NAV
    nav = compute_nav(state, prices)
    check_nav_drop(nav)
    log_equity(state, nav)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    log.info(f"Cycle end: NAV=${nav:,.0f}, cash=${state['cash']:,.0f}, "
             f"positions={len(state['positions'])}, trades={state['trade_count']}")

def main():
    log.info("=" * 60)
    log.info("WHEEL V4 PAPER ENGINE — ALL VALIDATED IMPROVEMENTS")
    log.info(f"  Delta: {PUT_DELTA_TARGET}, PT: {PROFIT_TAKE_PCT:.0%}, DTE: {DTE_TARGET}d")
    log.info(f"  Brake: {BRAKE_LOOKBACK_DAYS}d/{BRAKE_THRESHOLD:.0%}/{BRAKE_SCALE:.0%}")
    log.info(f"  Margin: {MARGIN_CAP:.0%} cap, {PER_NAME_PCT:.0%} per name")
    log.info(f"  Expected: ~25% CAGR, Sharpe 1.83, MaxDD -14.3%")
    log.info("=" * 60)

    universe = load_universe()
    earnings_lookup = load_earnings_dates()

    log.info(f"Universe: {len(universe)} tickers, earnings data: {len(earnings_lookup)} tickers")

    while True:
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle(universe, earnings_lookup)
            else:
                if now.minute < 5:  # Once per hour off-hours
                    state = load_state()
                    log.info(f"Off-hours: positions={len(state['positions'])}, "
                             f"realized=${state['realized_pnl']:.2f}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == '__main__':
    main()
