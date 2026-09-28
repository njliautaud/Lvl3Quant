#!/usr/bin/env python3
"""
Diversified Wheel Paper Engine — HC #660
==========================================

Paper-trades a 15-name diversified wheel basket selected from the
197-name universe sweep. Picks top Sharpe names across sectors.

Runs every 5 minutes during market hours, checks positions, opens
new CSPs, manages assignments, sells covered calls.

Uses yfinance for live prices, Black-Scholes for option pricing.

Author: Claude (HC #660 wheel expansion)
"""

import json
import math
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

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
STATE_DIR = ROOT / "live_trading_linux" / "wheel_diversified_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
EQUITY_FILE = STATE_DIR / "equity.csv"
TRADES_FILE = STATE_DIR / "trades.jsonl"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-DIV] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "wheel_diversified_paper.log")),
    ],
)
log = logging.getLogger('WHEEL-DIV')

# ── Configuration ──
STARTING_CAPITAL = 100_000.0
POLL_INTERVAL_SEC = 300  # 5 minutes

# Diversified basket: top Sharpe names per sector from 197-name WEEKLY sweep
# Quality-filtered: Sharpe >= 1.0, AnnRet > 0%, WR >= 60%. Max 3 per sector.
# Updated 2026-07-03 with weekly DTE (14d) backtest results.
BASKET = {
    # Ticker: (sector, weekly_backtest_sharpe)
    'WYNN': ('Consumer Cyclical', 1.77),
    'EXC':  ('Utilities', 1.71),
    'XOM':  ('Energy', 1.69),
    'TXN':  ('Technology', 1.65),
    'EA':   ('Communication Services', 1.64),
    'IBM':  ('Technology', 1.61),
    'DLR':  ('Real Estate', 1.61),
    'AEP':  ('Utilities', 1.61),
    'GILD': ('Healthcare', 1.60),
    'CAT':  ('Industrials', 1.60),
    'VZ':   ('Communication Services', 1.57),
    'CL':   ('Consumer Defensive', 1.56),
    'TMUS': ('Communication Services', 1.48),
    'IRM':  ('Real Estate', 1.46),
    'CVS':  ('Healthcare', 1.45),
    'AXP':  ('Financial Services', 1.43),
    'DUK':  ('Utilities', 1.41),
    'ABT':  ('Healthcare', 1.40),
    'CVX':  ('Energy', 1.38),
    'PG':   ('Consumer Defensive', 1.34),
    'HON':  ('Industrials', 1.33),
    'CSCO': ('Technology', 1.31),
    'HD':   ('Consumer Cyclical', 1.30),
    'VLO':  ('Energy', 1.29),
    'SBUX': ('Consumer Cyclical', 1.28),
    'GS':   ('Financial Services', 1.24),
    'TGT':  ('Consumer Defensive', 1.24),
    'SPG':  ('Real Estate', 1.22),
    'UNP':  ('Industrials', 1.16),
    'V':    ('Financial Services', 1.16),
    'LIN':  ('Basic Materials', 1.10),
}

# ── Sigma (volatility) filter ──
SIGMA_MAX_ENTRY = 0.8   # Skip tickers with 20d annualized vol > 80%

# ── Persistent loser blacklist ──
LOSER_BLACKLIST = {"CRSP", "AAL", "CELH"}

MAX_PER_NAME_PCT = 0.10  # Max 10% of NAV per name
MARGIN_CAP_PCT = 0.40     # Max 40% of NAV in total margin/collateral (HC #679 fix)
MAX_CONCURRENT = 15       # Max 15 simultaneous positions
PUT_DELTA_TARGET = 0.25
CALL_DELTA_TARGET = 0.30
DTE_TARGET = 14   # HC #660 param sweep: weekly optimal (+13% Sharpe vs monthly)
DTE_MIN = 10
DTE_MAX = 18
PROFIT_TAKE_PCT = 0.65     # 65% profit take (raised from 50% — peak profit study showed leaving 32% on table)
CSP_STOP_LOSS_MULT = 1.0   # Close CSP when loss reaches 1x premium collected
VIX_MAX_GATE = 35.0
DD_HALT_THRESHOLD = -0.07  # 3-day return < -7% halts new entries
CIRCUIT_BREAKER_PCT = -0.03  # 3% daily loss freezes entries for 1 day

RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

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

def strike_from_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return S
    p = (1.0 - abs(target_delta)) if kind == "put" else abs(target_delta)
    p = min(max(p, 1e-9), 1 - 1e-9)
    # Rational approximation to inverse normal CDF
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_ = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
          3.754408661907416]
    pl, pu = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    elif p <= pu:
        q = p - 0.5
        rr = q*q
        z = (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
    else:
        q = math.sqrt(-2 * math.log(1-p))
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    d1 = z
    K = S * math.exp((r + 0.5 * sigma**2) * T - d1 * sigma * math.sqrt(T))
    return max(0.01, round(K, 2))

# ── Price data ──
def get_live_prices(tickers):
    """Fetch current prices and 20d realized vol via yfinance."""
    import yfinance as yf
    prices = {}
    sigmas = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='60d')
            if hist.empty:
                continue
            prices[ticker] = float(hist['Close'].iloc[-1])
            log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            if len(log_ret) >= 20:
                sigmas[ticker] = float(log_ret.tail(20).std() * np.sqrt(252))
            else:
                sigmas[ticker] = float(log_ret.std() * np.sqrt(252))
            sigmas[ticker] = max(0.05, min(sigmas[ticker], 2.0))
        except Exception as e:
            log.warning(f"Price fetch failed for {ticker}: {e}")
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
    return 20.0  # default

def is_bear_regime():
    """SPY below 50d SMA = bear regime. Used for liq_csp_only gate.
    In bear: close CSPs early (no new ones), but keep shares + covered calls.
    Research showed this cuts bear drawdown 70% and improves Sharpe +0.09."""
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
    return False  # default to non-bear

# ── State management ──
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': STARTING_CAPITAL,
        'positions': [],  # list of position dicts
        'realized_pnl': 0.0,
        'trade_count': 0,
        'start_date': datetime.utcnow().isoformat(),
        'last_check': None,
    }

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def log_equity(state, prices):
    """Append to equity CSV."""
    nav = state['cash']
    unrealized = 0.0
    for pos in state['positions']:
        ticker = pos['ticker']
        if ticker not in prices:
            continue
        S = prices[ticker]
        if pos['side'] == 'short_put':
            T = max((pd.Timestamp(pos['expiry']) - pd.Timestamp.now()).days, 0) / 365
            sigma = pos.get('sigma', 0.20)
            current_premium = bs_price(S, pos['strike'], T, sigma, kind='put')
            unrealized += (pos['entry_premium'] - current_premium) * 100 * pos['contracts']
        elif pos['side'] == 'long_shares':
            unrealized += (S - pos['share_basis']) * 100 * pos['contracts']
        elif pos['side'] == 'short_call':
            T = max((pd.Timestamp(pos['expiry']) - pd.Timestamp.now()).days, 0) / 365
            sigma = pos.get('sigma', 0.20)
            current_premium = bs_price(S, pos['strike'], T, sigma, kind='call')
            unrealized += (pos['entry_premium'] - current_premium) * 100 * pos['contracts']
            unrealized += (S - pos['share_basis']) * 100 * pos['contracts']

    nav += unrealized
    row = f"{datetime.utcnow().isoformat()},{nav:.2f},{state['realized_pnl']:.2f},{unrealized:.2f}\n"

    if not EQUITY_FILE.exists():
        with open(EQUITY_FILE, 'w') as f:
            f.write("timestamp,nav,realized_pnl,unrealized_pnl\n")
    with open(EQUITY_FILE, 'a') as f:
        f.write(row)

    return nav

def log_trade(trade_info):
    with open(TRADES_FILE, 'a') as f:
        f.write(json.dumps(trade_info, default=str) + '\n')

# ── Trading logic ──
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
        expiry = pd.Timestamp(pos['expiry'])
        T = max((expiry - now).days, 0) / 365

        if pos['side'] == 'short_put':
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

            # Profit take
            if profit_pct >= PROFIT_TAKE_PCT:
                close_cost = current * 100 * pos['contracts'] + COST_PER_CONTRACT * pos['contracts']
                pnl = (pos['entry_premium'] - current) * 100 * pos['contracts'] - 2 * COST_PER_CONTRACT * pos['contracts']
                state['cash'] += pnl
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"CLOSE CSP {ticker} @ {profit_pct:.0%} profit. PnL: ${pnl:.2f}")
                log_trade({'action': 'close_csp', 'ticker': ticker, 'pnl': pnl,
                          'profit_pct': profit_pct, 'time': now.isoformat()})
                continue

            # Stop-loss: close CSP when loss reaches 1x premium collected
            if profit_pct <= -CSP_STOP_LOSS_MULT:
                close_cost = current * 100 * pos['contracts'] + COST_PER_CONTRACT * pos['contracts']
                pnl = (pos['entry_premium'] - current) * 100 * pos['contracts'] - 2 * COST_PER_CONTRACT * pos['contracts']
                state['cash'] += pnl
                state['realized_pnl'] += pnl
                state['trade_count'] += 1
                log.info(f"STOP LOSS CSP {ticker} @ {profit_pct:.0%} (loss >= {CSP_STOP_LOSS_MULT:.0f}x premium). PnL: ${pnl:.2f}")
                log_trade({'action': 'stop_loss_csp', 'ticker': ticker, 'pnl': pnl,
                          'profit_pct': round(profit_pct, 3), 'time': now.isoformat()})
                continue

            # Expiry
            if now >= expiry:
                if S < pos['strike']:
                    # Assignment — take shares
                    share_cost = pos['strike'] * 100 * pos['contracts']
                    state['cash'] -= share_cost
                    new_positions.append({
                        'ticker': ticker,
                        'side': 'long_shares',
                        'strike': 0,
                        'share_basis': pos['strike'] - pos['entry_premium'],
                        'expiry': '',
                        'entry_premium': 0,
                        'contracts': pos['contracts'],
                        'sigma': sigma,
                        'entry_date': now.isoformat(),
                    })
                    log.info(f"ASSIGNED {ticker} at {pos['strike']:.2f}. Basis: ${pos['strike'] - pos['entry_premium']:.2f}")
                    log_trade({'action': 'assigned', 'ticker': ticker,
                              'strike': pos['strike'], 'time': now.isoformat()})
                else:
                    # Expires OTM — keep premium
                    pnl = pos['entry_premium'] * 100 * pos['contracts'] - COST_PER_CONTRACT * pos['contracts']
                    state['realized_pnl'] += pnl
                    state['trade_count'] += 1
                    log.info(f"EXPIRED OTM {ticker} CSP. Premium kept: ${pnl:.2f}")
                    log_trade({'action': 'expired_otm', 'ticker': ticker,
                              'pnl': pnl, 'time': now.isoformat()})
                continue

            new_positions.append(pos)

        elif pos['side'] == 'long_shares':
            # Need to sell a covered call
            expiry_date = find_expiry()
            if expiry_date is None:
                new_positions.append(pos)
                continue

            T_new = (expiry_date - datetime.utcnow()).days / 365
            K = strike_from_delta(S, T_new, sigma, CALL_DELTA_TARGET, kind='call')
            # Use Alpaca real pricing when available, fall back to BS
            if _HAS_PRICING_BRIDGE:
                premium = _alpaca_get_premium(
                    ticker, S, K, T_new, sigma, expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date,
                    kind='call', bs_price_fn=bs_price)
            else:
                premium = bs_price(S, K, T_new, sigma, kind='call')
            slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)
            net_premium = max(0.01, premium - slip)

            state['cash'] += net_premium * 100 * pos['contracts'] - COST_PER_CONTRACT * pos['contracts']
            new_positions.append({
                'ticker': ticker,
                'side': 'short_call',
                'strike': K,
                'share_basis': pos['share_basis'],
                'expiry': expiry_date.isoformat(),
                'entry_premium': net_premium,
                'contracts': pos['contracts'],
                'sigma': sigma,
                'entry_date': now.isoformat(),
            })
            log.info(f"SELL CC {ticker} {K:.2f} strike, {(expiry_date - datetime.utcnow()).days}d, premium ${net_premium:.2f}")
            log_trade({'action': 'sell_cc', 'ticker': ticker, 'strike': K,
                      'premium': net_premium, 'time': now.isoformat()})

        elif pos['side'] == 'short_call':
            # FIX (HC #709 audit): Use consistent pricing — same source as entry
            expiry_date_cc = expiry.date() if hasattr(expiry, 'date') else expiry
            current = _get_option_price_consistent(ticker, S, pos['strike'], T, sigma, expiry_date_cc, kind='call')
            profit_pct = (pos['entry_premium'] - current) / pos['entry_premium'] if pos['entry_premium'] > 0 else 0

            # Profit take
            if profit_pct >= PROFIT_TAKE_PCT:
                pnl_cc = (pos['entry_premium'] - current) * 100 * pos['contracts'] - 2 * COST_PER_CONTRACT * pos['contracts']
                state['cash'] += pnl_cc
                state['realized_pnl'] += pnl_cc
                # Still have shares — go back to long_shares
                new_positions.append({
                    'ticker': ticker,
                    'side': 'long_shares',
                    'strike': 0,
                    'share_basis': pos['share_basis'],
                    'expiry': '',
                    'entry_premium': 0,
                    'contracts': pos['contracts'],
                    'sigma': sigma,
                    'entry_date': now.isoformat(),
                })
                log.info(f"CLOSE CC {ticker} @ {profit_pct:.0%} profit. PnL: ${pnl_cc:.2f}")
                continue

            # Expiry
            if now >= expiry:
                if S > pos['strike']:
                    # Called away — sell shares
                    sale = pos['strike'] * 100 * pos['contracts']
                    pnl = (pos['strike'] - pos['share_basis']) * 100 * pos['contracts']
                    pnl += pos['entry_premium'] * 100 * pos['contracts']
                    state['cash'] += sale
                    state['realized_pnl'] += pnl
                    state['trade_count'] += 1
                    log.info(f"CALLED AWAY {ticker} at {pos['strike']:.2f}. Full-cycle PnL: ${pnl:.2f}")
                    log_trade({'action': 'called_away', 'ticker': ticker,
                              'pnl': pnl, 'time': now.isoformat()})
                else:
                    # CC expires OTM — keep premium, keep shares
                    pnl = pos['entry_premium'] * 100 * pos['contracts'] - COST_PER_CONTRACT * pos['contracts']
                    state['realized_pnl'] += pnl
                    new_positions.append({
                        'ticker': ticker,
                        'side': 'long_shares',
                        'strike': 0,
                        'share_basis': pos['share_basis'],
                        'expiry': '',
                        'entry_premium': 0,
                        'contracts': pos['contracts'],
                        'sigma': sigma,
                        'entry_date': now.isoformat(),
                    })
                    log.info(f"CC EXPIRED OTM {ticker}. Premium kept: ${pnl:.2f}")
                continue

            new_positions.append(pos)

    state['positions'] = new_positions

def open_new_positions(state, prices, sigmas, vix, bear_mode=False, panic_scale=1.0):
    """Open CSPs on names without positions if capital allows.

    Args:
        panic_scale: multiplier from panic confluence monitor.
                     >1.0 = offensive (sell more premium), <1.0 = cautious.
    """
    if bear_mode:
        log.info("BEAR REGIME (SPY < 50d SMA) — no new CSPs (liq_csp_only mode)")
        return
    if vix > VIX_MAX_GATE:
        log.info(f"VIX {vix:.1f} > {VIX_MAX_GATE} gate — no new entries")
        return

    # Which tickers already have positions?
    active_tickers = {p['ticker'] for p in state['positions']}

    # Calculate NAV for position sizing
    nav = state['cash']
    for pos in state['positions']:
        if pos['side'] in ('long_shares', 'short_call'):
            ticker = pos['ticker']
            if ticker in prices:
                nav += prices[ticker] * 100 * pos['contracts']

    # ── Portfolio-level risk gates (HC #679 fix) ──
    # Max concurrent positions
    csp_count = sum(1 for p in state['positions'] if p['side'] == 'short_put')
    if csp_count >= MAX_CONCURRENT:
        log.info(f"MAX_CONCURRENT reached ({csp_count}/{MAX_CONCURRENT}) — no new entries")
        return

    # Portfolio margin cap — total collateral must not exceed MARGIN_CAP_PCT of NAV
    total_collateral = sum(
        p.get('strike', 0) * 100 * p.get('contracts', 1)
        for p in state['positions'] if p['side'] == 'short_put'
    )
    # Apply panic scale to margin cap (offensive = allow more margin, cautious = tighter)
    effective_margin_cap = min(MARGIN_CAP_PCT * panic_scale, 0.60)  # Hard ceiling: never exceed 60%
    margin_ratio = total_collateral / max(nav, 1)
    if margin_ratio >= effective_margin_cap:
        log.info(f"MARGIN CAP reached ({margin_ratio:.1%} >= {effective_margin_cap:.0%} of NAV "
                 f"[panic_scale={panic_scale:.2f}]) — no new entries")
        return

    # Drawdown halt — check 3-day return from equity curve
    eq = state.get('equity_curve', [])
    if len(eq) >= 3:
        nav_3d_ago = eq[-3].get('nav', nav) if isinstance(eq[-3], dict) else eq[-3]
        ret_3d = (nav - nav_3d_ago) / max(nav_3d_ago, 1)
        if ret_3d < DD_HALT_THRESHOLD:
            log.warning(f"DD HALT: 3-day return {ret_3d:.1%} < {DD_HALT_THRESHOLD:.0%} — no new entries")
            return

    # Circuit breaker — check daily return
    if len(eq) >= 1:
        nav_1d_ago = eq[-1].get('nav', nav) if isinstance(eq[-1], dict) else eq[-1]
        ret_1d = (nav - nav_1d_ago) / max(nav_1d_ago, 1)
        if ret_1d < CIRCUIT_BREAKER_PCT:
            log.warning(f"CIRCUIT BREAKER: daily return {ret_1d:.1%} < {CIRCUIT_BREAKER_PCT:.0%} — no new entries")
            return

    max_per_name = nav * MAX_PER_NAME_PCT * panic_scale

    for ticker, (sector, _) in BASKET.items():
        if ticker in active_tickers:
            continue
        if ticker not in prices or ticker not in sigmas:
            continue

        S = prices[ticker]
        sigma = sigmas[ticker]

        # Blacklist: consistent losers across engines
        if ticker in LOSER_BLACKLIST:
            continue

        # Sigma filter: skip extreme-vol names (sigma > 0.8 annualized)
        if sigma > SIGMA_MAX_ENTRY:
            log.info(f"SIGMA SKIP: {ticker} sigma={sigma:.2f} > {SIGMA_MAX_ENTRY:.2f} threshold")
            continue

        # HC #750: Multi-signal confluence check (min 2 confirming signals)
        try:
            from live_trading_linux.signal_confluence import check_confluence, log_confluence
            confluence = check_confluence(
                ticker, direction='short_put', price=S, vix=vix, sigma=sigma
            )
            if not confluence['pass']:
                log.debug(f"CONFLUENCE SKIP: {ticker} — {confluence['score']}/{confluence['min_required']} signals")
                continue
            log.info(f"CONFLUENCE PASS: {ticker} — {confluence['score']} signals: "
                     f"{', '.join(confluence['confirming_signals'])}")
        except ImportError:
            pass
        except Exception as _e:
            log.debug(f"Confluence check error for {ticker}: {_e}")

        # Can we afford 1 contract? (100 shares at strike)
        expiry_date = find_expiry()
        if expiry_date is None:
            continue
        T = (expiry_date - datetime.utcnow()).days / 365
        K = strike_from_delta(S, T, sigma, PUT_DELTA_TARGET, kind='put')

        collateral_needed = K * 100  # CSP requires cash to cover assignment
        if collateral_needed > max_per_name or collateral_needed > state['cash'] * 0.90:
            continue
        # Check adding this position wouldn't breach portfolio margin cap
        if (total_collateral + collateral_needed) / max(nav, 1) > effective_margin_cap:
            log.info(f"Skipping {ticker} — would breach margin cap ({(total_collateral + collateral_needed)/nav:.1%})")
            continue

        # Use Alpaca real pricing when available, fall back to BS
        if _HAS_PRICING_BRIDGE:
            premium = _alpaca_get_premium(
                ticker, S, K, T, sigma, expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date,
                kind='put', bs_price_fn=bs_price)
        else:
            premium = bs_price(S, K, T, sigma, kind='put')
        slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)
        net_premium = max(0.01, premium - slip)

        # Min premium filter: at least $0.20/share ($20/contract)
        if net_premium < 0.20:
            continue

        # Open position
        state['cash'] += net_premium * 100 - COST_PER_CONTRACT
        state['positions'].append({
            'ticker': ticker,
            'side': 'short_put',
            'strike': K,
            'share_basis': 0,
            'expiry': expiry_date.isoformat(),
            'entry_premium': net_premium,
            'contracts': 1,
            'sigma': sigma,
            'entry_date': datetime.utcnow().isoformat(),
        })

        dte = (expiry_date - datetime.utcnow()).days
        log.info(f"SELL CSP {ticker} {K:.2f} strike ({dte}d), premium ${net_premium:.2f}/sh, "
                 f"yield {net_premium/K*100:.1f}% ({sector})")
        log_trade({'action': 'sell_csp', 'ticker': ticker, 'strike': K,
                  'premium': net_premium, 'dte': dte, 'sigma': sigma,
                  'time': datetime.utcnow().isoformat()})

        # HC #702 — log option pricing for audit
        if _HAS_PRICING_LOGGER:
            log_option_price(
                engine="wheel-diversified", ticker=ticker, spot=S, strike=K,
                expiry=expiry_date.isoformat(), option_type="put", iv_used=sigma,
                bs_price=bs_price(S, K, T, sigma, kind='put'), source="bs", action="entry"
            )

# ── Main loop ──
def run_cycle():
    """Run one full check cycle."""
    state = load_state()

    log.info(f"Cycle start: cash=${state['cash']:.2f}, positions={len(state['positions'])}, "
             f"realized=${state['realized_pnl']:.2f}")

    # Fetch data
    tickers = list(BASKET.keys())
    prices, sigmas = get_live_prices(tickers)
    vix = get_vix()

    bear_mode = is_bear_regime()
    log.info(f"Prices for {len(prices)}/{len(tickers)} tickers, VIX={vix:.1f}, regime={'BEAR' if bear_mode else 'BULL'}")

    # In bear mode: close open CSPs early (liq_csp_only — keep shares + CCs)
    if bear_mode:
        csp_positions = [p for p in state['positions'] if p['side'] == 'short_put']
        for pos in csp_positions:
            ticker = pos['ticker']
            if ticker not in prices:
                continue
            S = prices[ticker]
            sigma = sigmas.get(ticker, pos.get('sigma', 0.20))
            expiry = pd.Timestamp(pos['expiry'])
            T = max((expiry - pd.Timestamp.now()).days, 0) / 365
            current = bs_price(S, pos['strike'], T, sigma, kind='put')
            pnl = (pos['entry_premium'] - current) * 100 * pos['contracts'] - 2 * COST_PER_CONTRACT * pos['contracts']
            state['cash'] += pnl
            state['realized_pnl'] += pnl
            state['trade_count'] += 1
            state['positions'].remove(pos)
            log.info(f"BEAR CLOSE CSP {ticker} (liq_csp_only). PnL: ${pnl:.2f}")
            log_trade({'action': 'bear_close_csp', 'ticker': ticker, 'pnl': pnl,
                      'time': pd.Timestamp.now().isoformat()})

    # Process existing positions
    process_positions(state, prices, sigmas)

    # ── Panic confluence overlay (offensive + defensive) ──
    panic_scale = 1.0
    try:
        from live_trading_linux.panic_confluence_monitor import get_panic_confluence
        panic = get_panic_confluence()
        panic_mult = panic.get('position_multiplier', 1.0)
        panic_mode = panic.get('mode', 'normal')
        if panic_mode == 'offensive':
            panic_scale = panic_mult
            log.info(f"PANIC OFFENSIVE: confluence {panic.get('confluence_score', 0)}/3, "
                     f"multiplier {panic_mult:.2f}x -> sell more premium")
        elif panic_mode == 'cautious':
            panic_scale = panic_mult  # 0.70x
            log.info(f"PANIC CAUTIOUS: stress building, "
                     f"multiplier {panic_mult:.2f}x -> tighter sizing")
    except Exception as e:
        log.debug(f"Panic confluence check skipped: {e}")

    # Open new positions (blocked in bear mode)
    open_new_positions(state, prices, sigmas, vix, bear_mode=bear_mode,
                       panic_scale=panic_scale)

    # Log equity
    nav = log_equity(state, prices)

    state['last_check'] = datetime.utcnow().isoformat()
    save_state(state)

    log.info(f"Cycle end: NAV=${nav:.2f}, cash=${state['cash']:.2f}, "
             f"positions={len(state['positions'])}, trades={state['trade_count']}")

def main():
    log.info("=" * 60)
    log.info("DIVERSIFIED WHEEL PAPER ENGINE — HC #660")
    log.info(f"Basket: {len(BASKET)} names across {len(set(s for s,_ in BASKET.values()))} sectors")
    log.info(f"Capital: ${STARTING_CAPITAL:,.0f}")
    log.info(f"Names: {', '.join(sorted(BASKET.keys()))}")
    log.info("=" * 60)

    # Initialize Alpaca real pricing (falls back to BS when unavailable)
    if _HAS_PRICING_BRIDGE:
        _init_bridge(engine_name='wheel-diversified')
        log.info("Alpaca real pricing bridge initialized")
    else:
        log.info("Alpaca pricing bridge not available, using BS-only pricing")

    while True:
        try:
            now = datetime.utcnow()
            # Run during extended hours (13:30-21:00 UTC = 9:30-17:00 ET, Mon-Fri)
            weekday = now.weekday()
            hour_utc = now.hour + now.minute / 60

            if weekday < 5 and 13.5 <= hour_utc <= 21.0:
                run_cycle()
            else:
                # Off-hours: just log equity snapshot less frequently
                if now.minute < 5:  # Once per hour off-hours
                    state = load_state()
                    prices, _ = get_live_prices(list(BASKET.keys()))
                    if prices:
                        nav = log_equity(state, prices)
                        log.info(f"Off-hours snapshot: NAV=${nav:.2f}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == '__main__':
    main()
