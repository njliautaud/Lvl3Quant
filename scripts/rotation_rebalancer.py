#!/usr/bin/env python3
"""
Vol-Adj Relative Strength Rotation Rebalancer — Agentic Account (RH $ROBINHOOD_ACCOUNT_NUMBER)
================================================================================
Implements the CHAMPION strategy (Sharpe 2.024, 6/6 adversarial).

Logic:
  1. Calculate 20-day risk-adjusted momentum (return/vol) for GLD, TLT, UUP
  2. Go 100% into the top-ranked asset
  3. Rebalance weekly (Fridays) or when earnings momentum signals override
  4. Kill switch: VIX > 20 AND SPY < 50-SMA → stay in GLD (safest haven)

Run: Fridays 10:00 AM ET via cron, or manually
  python3 scripts/rotation_rebalancer.py
  python3 scripts/rotation_rebalancer.py --status   # show current signal without trading
  python3 scripts/rotation_rebalancer.py --force     # force rebalance even if not Friday

Output: state/rotation_state.json, logs/rotation_rebalancer.log
"""

import json
import logging
import os
import sys
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_FILE = ROOT / "state" / "rotation_state.json"
LOG_FILE = ROOT / "logs" / "rotation_rebalancer.log"
EARNINGS_STATE = ROOT / "state" / "earnings_momentum_state.json"

ROTATION_ASSETS = ["GLD", "TLT", "UUP"]
LOOKBACK_DAYS = 20  # risk-adjusted momentum lookback
VOL_TARGET = 0.08   # 8% annualized vol target (for vol-adjusting)
ACCOUNT_NUMBER = os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", "")

# Kill switch thresholds
VIX_THRESHOLD = 20.0
SPY_SMA_PERIOD = 50

os.makedirs(ROOT / "state", exist_ok=True)
os.makedirs(ROOT / "logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("rotation")


# ── State Management ──────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "current_asset": None,
        "current_shares": 0,
        "entry_price": 0,
        "entry_date": None,
        "account_equity": 670.0,
        "rebalance_history": [],
        "last_rebalance": None,
        "mode": "paper",  # "paper" or "live"
    }


def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Market Data ───────────────────────────────────────────────────────────

def get_prices(tickers, period="90d"):
    """Fetch daily close prices."""
    data = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period=period)
            if len(hist) > 0:
                data[t] = hist["Close"]
        except Exception as e:
            log.warning(f"Failed to fetch {t}: {e}")
    return data


def calc_risk_adj_momentum(prices, lookback=LOOKBACK_DAYS):
    """Risk-adjusted momentum = return / volatility over lookback period."""
    if len(prices) < lookback + 1:
        return -999
    recent = prices.iloc[-lookback:]
    daily_rets = recent.pct_change().dropna()
    if len(daily_rets) < 5:
        return -999
    total_ret = (1 + daily_rets).prod() - 1
    vol = daily_rets.std() * np.sqrt(252)
    if vol < 1e-8:
        return 0
    return total_ret / vol


def check_kill_switch():
    """Check VIX + SPY kill switch. Returns (active, severity, vix, spy_price, spy_sma)."""
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        vix = float(vix_data["Close"].iloc[-1]) if len(vix_data) else 0
    except:
        vix = 0

    try:
        spy_data = yf.Ticker("SPY").history(period=f"{SPY_SMA_PERIOD + 10}d")
        spy_price = float(spy_data["Close"].iloc[-1]) if len(spy_data) else 0
        spy_sma = float(spy_data["Close"].tail(SPY_SMA_PERIOD).mean()) if len(spy_data) >= SPY_SMA_PERIOD else spy_price
    except:
        spy_price, spy_sma = 0, 0

    vix_high = vix > VIX_THRESHOLD
    spy_below_sma = spy_price < spy_sma

    if vix_high and spy_below_sma:
        return True, "FULL_STOP", vix, spy_price, spy_sma
    elif vix_high or spy_below_sma:
        return True, "HALF_SIZE", vix, spy_price, spy_sma
    else:
        return False, "CLEAR", vix, spy_price, spy_sma


# ── Signal Generation ─────────────────────────────────────────────────────

def generate_rotation_signal():
    """
    Calculate risk-adjusted momentum for GLD/TLT/UUP.
    Returns (best_asset, scores_dict, prices_dict).
    """
    prices = get_prices(ROTATION_ASSETS + ["SPY"], period="90d")

    scores = {}
    current_prices = {}
    for asset in ROTATION_ASSETS:
        if asset in prices and len(prices[asset]) > LOOKBACK_DAYS:
            scores[asset] = calc_risk_adj_momentum(prices[asset])
            current_prices[asset] = float(prices[asset].iloc[-1])
        else:
            scores[asset] = -999
            current_prices[asset] = 0

    best_asset = max(scores, key=scores.get)

    return best_asset, scores, current_prices


def check_earnings_override():
    """
    Check if earnings momentum scanner has active signals that should
    override the rotation (free up capital for burst trades).
    """
    if not EARNINGS_STATE.exists():
        return False, []

    try:
        with open(EARNINGS_STATE) as f:
            earnings = json.load(f)
        positions = earnings.get("positions", [])
        active = [p for p in positions if p.get("days_remaining", 0) > 0]
        return len(active) > 0, active
    except:
        return False, []


# ── Main Rebalance Logic ──────────────────────────────────────────────────

def run_rebalancer(force=False, status_only=False):
    state = load_state()
    today = date.today()
    is_friday = today.weekday() == 4

    log.info("=" * 60)
    log.info("VOL-ADJ RELATIVE STRENGTH ROTATION — Rebalancer")
    log.info(f"Date: {today} | Mode: {state.get('mode', 'paper')}")

    # 1. Kill switch check
    ks_active, ks_severity, vix, spy_price, spy_sma = check_kill_switch()
    log.info(f"Kill switch: {ks_severity} | VIX={vix:.1f} | SPY=${spy_price:.2f} (SMA50=${spy_sma:.2f})")

    # 2. Generate rotation signal
    best_asset, scores, current_prices = generate_rotation_signal()
    log.info(f"Risk-adj momentum scores:")
    for asset in ROTATION_ASSETS:
        marker = " ← BEST" if asset == best_asset else ""
        log.info(f"  {asset}: {scores[asset]:.4f} @ ${current_prices[asset]:.2f}{marker}")

    # 3. Kill switch override
    if ks_severity == "FULL_STOP":
        log.info("KILL SWITCH FULL_STOP — forcing GLD (safest haven)")
        best_asset = "GLD"
    elif ks_severity == "HALF_SIZE":
        log.info(f"KILL SWITCH HALF_SIZE — proceeding with {best_asset} but at reduced size")

    # 4. Check earnings override
    has_earnings, earnings_positions = check_earnings_override()
    if has_earnings:
        log.info(f"EARNINGS MOMENTUM ACTIVE: {len(earnings_positions)} position(s)")
        for ep in earnings_positions:
            log.info(f"  {ep.get('symbol')}: {ep.get('days_remaining', '?')} days remaining")

    # 5. Current state
    current_asset = state.get("current_asset")
    current_shares = state.get("current_shares", 0)
    equity = state.get("account_equity", 670)

    log.info(f"Current: {current_asset} ({current_shares} shares) | Equity: ${equity:.2f}")

    # 6. Determine action
    needs_rebalance = (current_asset != best_asset) and (is_friday or force)

    if status_only:
        log.info(f"STATUS ONLY — Signal: {best_asset} | Current: {current_asset} | Rebalance needed: {needs_rebalance}")
        if needs_rebalance:
            price = current_prices.get(best_asset, 0)
            if price > 0:
                available = equity * (0.5 if ks_severity == "HALF_SIZE" else 0.95)
                shares = available / price
                log.info(f"  WOULD: Sell {current_asset}, Buy {shares:.3f} shares {best_asset} @ ${price:.2f}")
        save_state(state)
        return state

    if not needs_rebalance:
        if current_asset == best_asset:
            log.info(f"HOLD — Already in {best_asset}. No rebalance needed.")
        elif not is_friday and not force:
            log.info(f"SIGNAL: {best_asset} but not Friday — holding {current_asset} until rebalance day.")
        save_state(state)
        return state

    # 7. Execute rebalance (paper mode: just update state)
    price = current_prices.get(best_asset, 0)
    if price <= 0:
        log.error(f"Cannot rebalance — {best_asset} price is ${price}")
        save_state(state)
        return state

    # Calculate position size
    if ks_severity == "HALF_SIZE":
        available = equity * 0.50
        log.info(f"HALF_SIZE: Using ${available:.2f} of ${equity:.2f}")
    else:
        available = equity * 0.95
        log.info(f"FULL SIZE: Using ${available:.2f} of ${equity:.2f}")

    new_shares = round(available / price, 4)

    # Record rebalance
    rebal_record = {
        "date": today.isoformat(),
        "from_asset": current_asset,
        "to_asset": best_asset,
        "shares": new_shares,
        "price": price,
        "amount": round(new_shares * price, 2),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "kill_switch": ks_severity,
        "vix": round(vix, 2),
    }

    state["current_asset"] = best_asset
    state["current_shares"] = new_shares
    state["entry_price"] = price
    state["entry_date"] = today.isoformat()
    state["last_rebalance"] = today.isoformat()
    state["rebalance_history"].append(rebal_record)

    # Keep only last 52 rebalances (1 year)
    if len(state["rebalance_history"]) > 52:
        state["rebalance_history"] = state["rebalance_history"][-52:]

    log.info(f"REBALANCE: {current_asset or 'CASH'} → {best_asset}")
    log.info(f"  Buy {new_shares:.4f} shares {best_asset} @ ${price:.2f} = ${new_shares * price:.2f}")

    if state.get("mode") == "live":
        log.info("LIVE MODE — generating RH order instructions")
        log.info(f"  1. Sell ALL {current_asset} on RH")
        log.info(f"  2. Buy ${new_shares * price:.2f} of {best_asset} on RH")
        # TODO: Auto-execute via Robinhood MCP when ready
    else:
        log.info("PAPER MODE — orders logged but not executed")

    save_state(state)
    log.info("Done.")
    return state


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation Rebalancer")
    parser.add_argument("--status", action="store_true", help="Show signal without rebalancing")
    parser.add_argument("--force", action="store_true", help="Force rebalance even if not Friday")
    args = parser.parse_args()

    run_rebalancer(force=args.force, status_only=args.status)
