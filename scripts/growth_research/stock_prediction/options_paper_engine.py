#!/usr/bin/env python3
"""
Options Paper Trading Engine
==============================
Simulates buying ATM calls based on daily scanner high-conviction signals.

Rules:
  - Enter when scanner confidence > 0.7 (HIGH CONVICTION only)
  - Buy ATM call, 45 DTE
  - Position size: $500 per trade (single contract equivalent)
  - Max 3 open positions at once
  - Exit at 50% profit (take profit)
  - Exit at 14 DTE (time exit — roll or close)
  - Exit at 100% loss (stop loss)
  - Starting NAV: $10,000
  - Commission-free (Robinhood)

Uses Black-Scholes for mark-to-market (no live options data).

Usage:
    python options_paper_engine.py               # Normal daily run
    python options_paper_engine.py --status       # Show current positions
    python options_paper_engine.py --reset        # Reset paper account
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

for pkg in ["yfinance"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import yfinance as yf

# --- Paths ---
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
SCAN_DIR = BASE_DIR / "output/growth_research/stock_prediction/daily_scans"
STATE_DIR = BASE_DIR / "data/paper_engines/options_prediction"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
STARTING_NAV = 10000.0
POSITION_SIZE = 500.0
MAX_POSITIONS = 3
DTE = 45
TAKE_PROFIT_PCT = 0.50    # Close at 50% gain
STOP_LOSS_PCT = -1.00     # Close at 100% loss
TIME_EXIT_DTE = 14        # Close at 14 DTE remaining
RISK_FREE_RATE = 0.05
BID_ASK_SPREAD_PCT = 0.05  # 5% bid-ask spread on option premium
CONFIDENCE_THRESHOLD = 0.7  # Only trade HIGH CONVICTION signals


# --- Black-Scholes ---
def bs_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# --- State Management ---
def load_state():
    """Load or initialize paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)

    state = {
        'created': datetime.now().isoformat(),
        'starting_nav': STARTING_NAV,
        'cash': STARTING_NAV,
        'positions': [],  # Open positions
        'nav_history': [{'date': datetime.now().strftime('%Y-%m-%d'), 'nav': STARTING_NAV, 'cash': STARTING_NAV}],
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
    }
    save_state(state)
    return state


def save_state(state):
    """Save state to disk."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_trades():
    """Load trade history."""
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    """Save trade history."""
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)


def get_current_price(ticker):
    """Get current stock price."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")
        if len(hist) > 0:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


def get_historical_vol(ticker, lookback=63):
    """Get historical volatility for BS pricing."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{lookback + 10}d")
        if len(hist) >= 20:
            returns = hist['Close'].pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            return float(vol)
    except Exception:
        pass
    return 0.30  # Default 30% vol if can't compute


def estimate_iv(hist_vol):
    """
    Estimate implied volatility from historical vol.
    IV is typically higher than HV (volatility risk premium).
    """
    return hist_vol * 1.20  # 20% premium over HV


# --- Core Engine ---
def open_position(state, trades, ticker, price, confidence, driver, scan_date):
    """Open a new paper options position."""
    if len(state['positions']) >= MAX_POSITIONS:
        print(f"[PAPER] SKIP {ticker}: max {MAX_POSITIONS} positions reached")
        return False

    # Check if already have position in this ticker
    for pos in state['positions']:
        if pos['ticker'] == ticker:
            print(f"[PAPER] SKIP {ticker}: already have open position")
            return False

    # Check cash
    if state['cash'] < POSITION_SIZE:
        print(f"[PAPER] SKIP {ticker}: insufficient cash (${state['cash']:.0f} < ${POSITION_SIZE:.0f})")
        return False

    # Get volatility for BS pricing
    hist_vol = get_historical_vol(ticker)
    iv = estimate_iv(hist_vol)

    # ATM call pricing
    S = price
    K = round(price, 0)  # Round to nearest dollar for ATM strike
    T = DTE / 365.0
    r = RISK_FREE_RATE

    theoretical_premium = bs_call(S, K, T, r, iv)
    if theoretical_premium <= 0.01:
        print(f"[PAPER] SKIP {ticker}: premium too low (${theoretical_premium:.2f})")
        return False

    # Pay ask (premium + half spread)
    entry_premium = theoretical_premium * (1 + BID_ASK_SPREAD_PCT / 2)

    # Per-share premium, 100 shares per contract
    cost_per_contract = entry_premium * 100

    # How many contracts for target position size
    n_contracts = max(1, int(POSITION_SIZE / cost_per_contract))
    total_cost = n_contracts * cost_per_contract

    # Ensure we can afford it
    if total_cost > state['cash']:
        n_contracts = max(1, int(state['cash'] / cost_per_contract))
        total_cost = n_contracts * cost_per_contract
        if total_cost > state['cash']:
            print(f"[PAPER] SKIP {ticker}: can't afford even 1 contract (${cost_per_contract:.0f})")
            return False

    delta = bs_delta(S, K, T, r, iv)

    position = {
        'ticker': ticker,
        'entry_date': scan_date,
        'entry_price': round(price, 2),
        'strike': K,
        'dte_at_entry': DTE,
        'expiry_date': (pd.Timestamp(scan_date) + timedelta(days=DTE)).strftime('%Y-%m-%d'),
        'iv': round(iv, 4),
        'entry_premium': round(entry_premium, 4),
        'n_contracts': n_contracts,
        'total_cost': round(total_cost, 2),
        'confidence': round(confidence, 4),
        'driver': driver,
        'current_value': round(total_cost, 2),
        'unrealized_pnl': 0.0,
        'unrealized_pnl_pct': 0.0,
        'delta': round(delta, 3),
        'last_mark_date': scan_date,
    }

    state['positions'].append(position)
    state['cash'] -= total_cost
    state['cash'] = round(state['cash'], 2)
    state['total_trades'] += 1

    print(f"[PAPER] OPENED: {ticker} {n_contracts}x ${K:.0f}C @ ${entry_premium:.2f} "
          f"(cost: ${total_cost:.0f}, conf: {confidence:.1%}, delta: {delta:.2f})")
    print(f"         Driver: {driver}")

    return True


def mark_to_market(state, current_date=None):
    """Mark all positions to market using Black-Scholes."""
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')

    today = pd.Timestamp(current_date)
    positions_to_close = []

    for i, pos in enumerate(state['positions']):
        ticker = pos['ticker']
        price = get_current_price(ticker)
        if price is None:
            print(f"[PAPER] WARN: Can't get price for {ticker}, keeping last mark")
            continue

        # Remaining time to expiry
        expiry = pd.Timestamp(pos['expiry_date'])
        days_remaining = max((expiry - today).days, 0)
        T_remaining = days_remaining / 365.0

        # Mark using BS
        K = pos['strike']
        iv = pos['iv']
        r = RISK_FREE_RATE

        if T_remaining > 0:
            current_premium = bs_call(price, K, T_remaining, r, iv)
        else:
            current_premium = max(price - K, 0)

        # Apply bid-ask for exit (we'd get bid)
        exit_premium = current_premium * (1 - BID_ASK_SPREAD_PCT / 2)
        current_value = pos['n_contracts'] * exit_premium * 100
        unrealized_pnl = current_value - pos['total_cost']
        unrealized_pnl_pct = unrealized_pnl / pos['total_cost'] if pos['total_cost'] > 0 else 0

        delta = bs_delta(price, K, T_remaining, r, iv)

        pos['current_value'] = round(current_value, 2)
        pos['unrealized_pnl'] = round(unrealized_pnl, 2)
        pos['unrealized_pnl_pct'] = round(unrealized_pnl_pct, 4)
        pos['delta'] = round(delta, 3)
        pos['last_mark_date'] = current_date
        pos['current_stock_price'] = round(price, 2)
        pos['days_remaining'] = days_remaining

        # Check exit conditions
        exit_reason = None
        if unrealized_pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = f"TAKE PROFIT ({unrealized_pnl_pct:.0%} >= {TAKE_PROFIT_PCT:.0%})"
        elif unrealized_pnl_pct <= STOP_LOSS_PCT:
            exit_reason = f"STOP LOSS ({unrealized_pnl_pct:.0%} <= {STOP_LOSS_PCT:.0%})"
        elif days_remaining <= TIME_EXIT_DTE:
            exit_reason = f"TIME EXIT ({days_remaining}d remaining <= {TIME_EXIT_DTE}d)"
        elif days_remaining <= 0:
            exit_reason = "EXPIRY"

        if exit_reason:
            positions_to_close.append((i, exit_reason, current_value, unrealized_pnl, price))

    return positions_to_close


def close_position(state, trades_log, pos_idx, reason, exit_value, pnl, exit_price):
    """Close a position and record the trade."""
    pos = state['positions'][pos_idx]

    trade_record = {
        'ticker': pos['ticker'],
        'entry_date': pos['entry_date'],
        'exit_date': datetime.now().strftime('%Y-%m-%d'),
        'entry_price': pos['entry_price'],
        'exit_stock_price': round(exit_price, 2),
        'stock_return': round((exit_price / pos['entry_price'] - 1), 4),
        'strike': pos['strike'],
        'entry_premium': pos['entry_premium'],
        'n_contracts': pos['n_contracts'],
        'total_cost': pos['total_cost'],
        'exit_value': round(exit_value, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl / pos['total_cost'], 4) if pos['total_cost'] > 0 else 0,
        'exit_reason': reason,
        'confidence': pos['confidence'],
        'driver': pos['driver'],
        'days_held': (pd.Timestamp(datetime.now().strftime('%Y-%m-%d')) -
                      pd.Timestamp(pos['entry_date'])).days,
    }

    trades_log.append(trade_record)

    # Update state
    state['cash'] += exit_value
    state['cash'] = round(state['cash'], 2)
    state['total_pnl'] += pnl
    state['total_pnl'] = round(state['total_pnl'], 2)

    if pnl > 0:
        state['winning_trades'] += 1
    else:
        state['losing_trades'] += 1

    print(f"[PAPER] CLOSED: {pos['ticker']} — {reason}")
    print(f"         P&L: ${pnl:+.2f} ({pnl / pos['total_cost']:.0%}) | "
          f"Stock: ${pos['entry_price']:.2f} -> ${exit_price:.2f} "
          f"({(exit_price / pos['entry_price'] - 1):.1%})")

    return trade_record


def run_daily(scan_date=None):
    """Run the daily paper trading cycle."""
    if scan_date is None:
        scan_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*70}")
    print(f"[PAPER] Options Paper Engine — {scan_date}")
    print(f"{'='*70}")

    state = load_state()
    trades_log = load_trades()

    print(f"[PAPER] NAV: ${state['cash'] + sum(p.get('current_value', 0) for p in state['positions']):.2f} | "
          f"Cash: ${state['cash']:.2f} | Open positions: {len(state['positions'])}")

    # Step 1: Mark existing positions to market
    if state['positions']:
        print(f"\n--- MARK TO MARKET ---")
        exits = mark_to_market(state, scan_date)

        # Show positions
        for pos in state['positions']:
            print(f"  {pos['ticker']}: ${pos.get('current_stock_price', pos['entry_price']):.2f} | "
                  f"P&L: ${pos['unrealized_pnl']:+.2f} ({pos['unrealized_pnl_pct']:.0%}) | "
                  f"Delta: {pos['delta']:.2f} | {pos.get('days_remaining', '?')}d left")

        # Step 2: Close positions that hit exit rules
        if exits:
            print(f"\n--- CLOSING POSITIONS ---")
            # Process in reverse order to maintain indices
            for pos_idx, reason, exit_val, pnl, exit_price in sorted(exits, reverse=True):
                close_position(state, trades_log, pos_idx, reason, exit_val, pnl, exit_price)

            # Remove closed positions (reverse order)
            for pos_idx, _, _, _, _ in sorted(exits, reverse=True):
                state['positions'].pop(pos_idx)

    # Step 3: Look for new signals from scanner
    scan_file = SCAN_DIR / f"scan_{scan_date}.json"
    if scan_file.exists():
        with open(scan_file) as f:
            scan_data = json.load(f)

        high_conviction = [s for s in scan_data.get('signals', [])
                           if s['confidence'] >= CONFIDENCE_THRESHOLD]

        if high_conviction:
            print(f"\n--- NEW SIGNALS ({len(high_conviction)} high conviction) ---")
            for signal in high_conviction:
                if len(state['positions']) >= MAX_POSITIONS:
                    print(f"[PAPER] Max positions reached, skipping remaining signals")
                    break

                price = get_current_price(signal['ticker'])
                if price is None:
                    price = signal['price']  # Use scanner price as fallback

                opened = open_position(
                    state, trades_log,
                    ticker=signal['ticker'],
                    price=price,
                    confidence=signal['confidence'],
                    driver=signal['driver'],
                    scan_date=scan_date,
                )
        else:
            print(f"\n[PAPER] No high conviction signals today")
    else:
        print(f"\n[PAPER] No scan results found for {scan_date}")

    # Step 4: Update NAV history
    total_position_value = sum(p.get('current_value', p['total_cost']) for p in state['positions'])
    nav = state['cash'] + total_position_value

    state['nav_history'].append({
        'date': scan_date,
        'nav': round(nav, 2),
        'cash': round(state['cash'], 2),
        'positions_value': round(total_position_value, 2),
        'n_positions': len(state['positions']),
    })

    # Print summary
    print(f"\n{'='*70}")
    print(f"PORTFOLIO SUMMARY — {scan_date}")
    print(f"{'='*70}")
    print(f"  NAV:            ${nav:,.2f} ({(nav / STARTING_NAV - 1):+.1%} from start)")
    print(f"  Cash:           ${state['cash']:,.2f}")
    print(f"  Positions:      {len(state['positions'])} / {MAX_POSITIONS}")
    print(f"  Total trades:   {state['total_trades']}")

    completed = state['winning_trades'] + state['losing_trades']
    if completed > 0:
        wr = state['winning_trades'] / completed
        print(f"  Win rate:       {wr:.0%} ({state['winning_trades']}W / {state['losing_trades']}L)")
        print(f"  Realized P&L:   ${state['total_pnl']:+,.2f}")

        # Risk metrics from trade history
        if len(trades_log) > 2:
            returns = [t['pnl_pct'] for t in trades_log]
            avg_ret = np.mean(returns)
            std_ret = np.std(returns) if len(returns) > 1 else 1
            if std_ret > 0:
                # Annualize assuming ~8 trades/year (45-day holds)
                trades_per_year = 365 / 45
                sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)
                print(f"  Sharpe (est):   {sharpe:.2f}")

                downside = [r for r in returns if r < 0]
                if downside:
                    sortino = (avg_ret / np.std(downside)) * np.sqrt(trades_per_year)
                    print(f"  Sortino (est):  {sortino:.2f}")

            wins = [t['pnl'] for t in trades_log if t['pnl'] > 0]
            losses = [t['pnl'] for t in trades_log if t['pnl'] < 0]
            if wins and losses:
                pf = sum(wins) / abs(sum(losses))
                print(f"  Profit Factor:  {pf:.2f}")
                print(f"  Avg Win:        ${np.mean(wins):+.2f}")
                print(f"  Avg Loss:       ${np.mean(losses):+.2f}")

    # Save
    save_state(state)
    save_trades(trades_log)
    print(f"\n[PAPER] State saved.")

    return state, trades_log


def show_status():
    """Show current portfolio status without making any changes."""
    state = load_state()
    trades_log = load_trades()

    total_pos_val = sum(p.get('current_value', p['total_cost']) for p in state['positions'])
    nav = state['cash'] + total_pos_val

    print(f"\n{'='*70}")
    print(f"OPTIONS PAPER PORTFOLIO STATUS")
    print(f"{'='*70}")
    print(f"  NAV:          ${nav:,.2f} ({(nav / STARTING_NAV - 1):+.1%} total return)")
    print(f"  Cash:         ${state['cash']:,.2f}")
    print(f"  Invested:     ${total_pos_val:,.2f}")
    print(f"  Realized P&L: ${state['total_pnl']:+,.2f}")

    if state['positions']:
        print(f"\n  OPEN POSITIONS ({len(state['positions'])}):")
        print(f"  {'Ticker':<8} {'Strike':>8} {'Entry':>8} {'Now':>8} {'P&L':>10} {'P&L%':>8} {'Days':>6}")
        print(f"  {'-'*60}")
        for pos in state['positions']:
            print(f"  {pos['ticker']:<8} "
                  f"${pos['strike']:>6.0f} "
                  f"${pos['entry_price']:>7.2f} "
                  f"${pos.get('current_stock_price', pos['entry_price']):>7.2f} "
                  f"${pos['unrealized_pnl']:>+9.2f} "
                  f"{pos['unrealized_pnl_pct']:>+7.0%} "
                  f"{pos.get('days_remaining', '?'):>5}")
    else:
        print(f"\n  No open positions.")

    completed = state['winning_trades'] + state['losing_trades']
    if completed > 0:
        wr = state['winning_trades'] / completed
        print(f"\n  TRADE HISTORY ({completed} closed):")
        print(f"  Win Rate: {wr:.0%} | Wins: {state['winning_trades']} | Losses: {state['losing_trades']}")

    if trades_log:
        print(f"\n  RECENT TRADES:")
        print(f"  {'Date':<12} {'Ticker':<8} {'P&L':>10} {'Reason':<20}")
        for t in trades_log[-5:]:
            print(f"  {t['exit_date']:<12} {t['ticker']:<8} ${t['pnl']:>+9.2f} {t['exit_reason']:<20}")

    return state


def reset_account():
    """Reset the paper trading account to starting state."""
    print("[PAPER] Resetting paper trading account...")
    state = {
        'created': datetime.now().isoformat(),
        'starting_nav': STARTING_NAV,
        'cash': STARTING_NAV,
        'positions': [],
        'nav_history': [{'date': datetime.now().strftime('%Y-%m-%d'), 'nav': STARTING_NAV, 'cash': STARTING_NAV}],
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
    }
    save_state(state)
    save_trades([])
    print(f"[PAPER] Account reset to ${STARTING_NAV:,.0f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Options Paper Trading Engine")
    parser.add_argument('--status', action='store_true', help="Show current positions")
    parser.add_argument('--reset', action='store_true', help="Reset paper account")
    parser.add_argument('--date', type=str, default=None, help="Run for specific date")
    args = parser.parse_args()

    if args.reset:
        reset_account()
    elif args.status:
        show_status()
    else:
        run_daily(args.date)


if __name__ == "__main__":
    main()
