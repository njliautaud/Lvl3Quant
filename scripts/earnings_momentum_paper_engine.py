#!/usr/bin/env python3
"""
Earnings Surprise Momentum — Paper Trading Engine
===================================================
Tracks the Beat-Chain variant (B) from our validated backtest:
- Sharpe 1.215, WR 55.2%, MDD -28.8%, perm p=0.004
- Buy shares after earnings beats on growth stocks
- Only enter if stock has beaten 2+ consecutive quarters
- Hold ~60 trading days (until ~5 days before next earnings)
- Kill switch: skip entries when VIX>20 AND SPY<50SMA

Run daily after market close to:
1. Check for new earnings beats (today's reporters)
2. Manage existing positions (exit at ~60 days or pre-next-earnings)
3. Update NAV and paper P&L

Usage: python3 earnings_momentum_paper_engine.py [--check-only]
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────
STATE_FILE = Path("/home/jupiter/Lvl3Quant/data/paper_engines/earnings_momentum/state.json")
TRADES_FILE = Path("/home/jupiter/Lvl3Quant/data/paper_engines/earnings_momentum/trades.json")

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER",
    "LYFT", "COIN", "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU"
]

INITIAL_CAPITAL = 10000.0  # Paper account size
MAX_POSITIONS = 5
HOLD_DAYS = 60  # Trading days
PRE_EARNINGS_EXIT = 5  # Exit N days before next earnings
MIN_GAP_PCT = 3.0  # Minimum gap to qualify as "beat"
MIN_CONSECUTIVE_BEATS = 2  # Require 2+ consecutive beats (Beat-Chain variant)
SLIPPAGE_BPS = 2  # 0.02% slippage per side


def load_state():
    """Load or initialize paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "nav": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "beat_history": {},  # {ticker: [list of beat dates]}
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "strategy": "earnings_surprise_momentum_beat_chain",
        "variant": "B",
        "backtest_sharpe": 1.215,
        "backtest_perm_p": 0.004,
    }


def load_trades():
    """Load trade history."""
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_state(state):
    """Save state to disk."""
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def save_trades(trades):
    """Save trade history."""
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def check_kill_switch():
    """Check if kill switch is active (VIX>20 AND SPY<50SMA)."""
    try:
        spy = yf.download("SPY", period="60d", progress=False)
        vix = yf.download("^VIX", period="5d", progress=False)

        if spy.empty or vix.empty:
            return True, "Data unavailable — defaulting to kill switch ON"

        # Handle multi-level columns
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.droplevel(1, axis=1)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix.droplevel(1, axis=1)

        spy_price = float(spy['Close'].iloc[-1])
        spy_50sma = float(spy['Close'].rolling(50).mean().iloc[-1])
        vix_level = float(vix['Close'].iloc[-1])

        active = vix_level > 20 and spy_price < spy_50sma
        reason = f"VIX={vix_level:.1f}, SPY={spy_price:.1f} vs 50SMA={spy_50sma:.1f}"
        return active, reason
    except Exception as e:
        return True, f"Error checking kill switch: {e}"


def detect_earnings_gaps(lookback_days=5):
    """
    Detect stocks that gapped >MIN_GAP_PCT in the last N trading days.
    Used as proxy for earnings beats.
    """
    gaps = []
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 10)  # Extra buffer

    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty or len(df) < 2:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            # Check last N trading days for gaps
            for i in range(-min(lookback_days, len(df)-1), 0):
                prev_close = float(df['Close'].iloc[i-1])
                open_price = float(df['Open'].iloc[i])
                gap_pct = (open_price - prev_close) / prev_close * 100

                if gap_pct >= MIN_GAP_PCT:
                    gap_date = df.index[i].strftime('%Y-%m-%d')
                    current_price = float(df['Close'].iloc[-1])
                    gaps.append({
                        "ticker": ticker,
                        "gap_date": gap_date,
                        "gap_pct": round(gap_pct, 2),
                        "gap_open": round(open_price, 2),
                        "current_price": round(current_price, 2),
                    })
        except Exception:
            continue

    return gaps


def check_consecutive_beats(state, ticker, new_beat_date):
    """Check if this ticker has consecutive beats (for Beat-Chain variant)."""
    history = state.get("beat_history", {}).get(ticker, [])

    if not history:
        # First beat — record but don't trade yet
        return False, 1

    # Check if last beat was within ~100 days (one quarter)
    last_beat = datetime.strptime(history[-1], '%Y-%m-%d')
    new_beat = datetime.strptime(new_beat_date, '%Y-%m-%d')
    days_since_last = (new_beat - last_beat).days

    if 60 <= days_since_last <= 120:
        # Consecutive beat (within a quarter)
        consecutive = len(history) + 1
        return consecutive >= MIN_CONSECUTIVE_BEATS, consecutive
    else:
        # Gap in beats — reset
        return False, 1


def update_beat_history(state, ticker, beat_date):
    """Add a beat to ticker's history."""
    if "beat_history" not in state:
        state["beat_history"] = {}
    if ticker not in state["beat_history"]:
        state["beat_history"][ticker] = []

    # Avoid duplicates
    if beat_date not in state["beat_history"][ticker]:
        state["beat_history"][ticker].append(beat_date)
        # Keep last 8 quarters
        state["beat_history"][ticker] = state["beat_history"][ticker][-8:]


def get_current_prices(tickers):
    """Get current prices for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="2d", progress=False)
        if data.empty:
            return prices
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    prices[ticker] = float(data['Close'][ticker].iloc[-1])
                except (KeyError, IndexError):
                    pass
        else:
            # Single ticker
            prices[tickers[0]] = float(data['Close'].iloc[-1])
    except Exception:
        pass
    return prices


def run_engine(check_only=False):
    """Main paper trading engine loop."""
    state = load_state()
    trades = load_trades()
    actions = []

    print(f"={'='*60}")
    print(f"EARNINGS SURPRISE MOMENTUM — Paper Engine (Beat-Chain B)")
    print(f"{'='*60}")
    print(f"NAV: ${state['nav']:.2f}  Cash: ${state['cash']:.2f}")
    print(f"Positions: {len(state['positions'])} / {MAX_POSITIONS}")
    print(f"Record: {state['wins']}W-{state['losses']}L  PnL: ${state['total_pnl']:.2f}")
    print()

    # 1. Check kill switch
    ks_active, ks_reason = check_kill_switch()
    print(f"Kill switch: {'🔴 ACTIVE' if ks_active else '🟢 OFF'} — {ks_reason}")

    # 2. Update existing positions
    if state['positions']:
        pos_tickers = list(state['positions'].keys())
        prices = get_current_prices(pos_tickers)

        today = datetime.now()
        exits = []

        for ticker, pos in state['positions'].items():
            entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            days_held = (today - entry_date).days
            entry_px = pos['entry_price']
            current_px = prices.get(ticker, entry_px)
            pnl_pct = (current_px - entry_px) / entry_px * 100
            pnl_dollars = pos['shares'] * (current_px - entry_px)

            should_exit = days_held >= HOLD_DAYS
            reason = ""
            if should_exit:
                reason = f"Hold period complete ({days_held}d)"

            status = "EXIT" if should_exit else "HOLD"
            print(f"  {ticker}: ${entry_px:.2f} → ${current_px:.2f} ({pnl_pct:+.1f}%) "
                  f"Day {days_held}/{HOLD_DAYS} [{status}]")

            if should_exit and not check_only:
                # Apply slippage
                exit_px = current_px * (1 - SLIPPAGE_BPS / 10000)
                realized_pnl = pos['shares'] * (exit_px - entry_px)

                exits.append(ticker)
                state['cash'] += pos['shares'] * exit_px
                state['total_pnl'] += realized_pnl
                state['total_trades'] += 1
                if realized_pnl > 0:
                    state['wins'] += 1
                else:
                    state['losses'] += 1

                trade_record = {
                    "ticker": ticker,
                    "entry_date": pos['entry_date'],
                    "exit_date": today.strftime('%Y-%m-%d'),
                    "entry_price": entry_px,
                    "exit_price": round(exit_px, 2),
                    "shares": pos['shares'],
                    "pnl_dollars": round(realized_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": days_held,
                    "reason": reason,
                }
                trades.append(trade_record)
                actions.append(f"EXIT {ticker}: ${realized_pnl:+.2f} ({pnl_pct:+.1f}%)")

        for ticker in exits:
            del state['positions'][ticker]

    # 3. Scan for new entries (if kill switch allows and we have capacity)
    open_slots = MAX_POSITIONS - len(state['positions'])

    if ks_active:
        print(f"\n⚠️  Kill switch ACTIVE — no new entries")
    elif open_slots <= 0:
        print(f"\n⚠️  Max positions reached ({MAX_POSITIONS}) — no new entries")
    else:
        print(f"\nScanning for earnings gaps (last 3 days)...")
        gaps = detect_earnings_gaps(lookback_days=3)

        if not gaps:
            print("  No qualifying gaps found")
        else:
            print(f"  Found {len(gaps)} gaps ≥ {MIN_GAP_PCT}%:")
            new_entries = []

            for gap in sorted(gaps, key=lambda x: x['gap_pct'], reverse=True):
                ticker = gap['ticker']

                # Skip if already holding
                if ticker in state['positions']:
                    print(f"    {ticker}: +{gap['gap_pct']}% gap on {gap['gap_date']} — ALREADY HOLDING")
                    continue

                # Check consecutive beats
                qualifies, streak = check_consecutive_beats(state, ticker, gap['gap_date'])
                update_beat_history(state, ticker, gap['gap_date'])

                if qualifies:
                    print(f"    {ticker}: +{gap['gap_pct']}% gap on {gap['gap_date']} — ✅ QUALIFIES (streak: {streak})")
                    if len(new_entries) < open_slots:
                        new_entries.append(gap)
                else:
                    print(f"    {ticker}: +{gap['gap_pct']}% gap on {gap['gap_date']} — "
                          f"streak only {streak} (need {MIN_CONSECUTIVE_BEATS})")

            # Execute new entries
            if new_entries and not check_only:
                alloc_per_position = state['cash'] / max(open_slots, len(new_entries))
                alloc_per_position = min(alloc_per_position, state['cash'] / len(new_entries))

                for gap in new_entries:
                    ticker = gap['ticker']
                    entry_px = gap['current_price'] * (1 + SLIPPAGE_BPS / 10000)
                    shares = alloc_per_position / entry_px

                    state['positions'][ticker] = {
                        "entry_date": datetime.now().strftime('%Y-%m-%d'),
                        "entry_price": round(entry_px, 2),
                        "shares": round(shares, 4),
                        "gap_date": gap['gap_date'],
                        "gap_pct": gap['gap_pct'],
                        "alloc_usd": round(alloc_per_position, 2),
                    }
                    state['cash'] -= shares * entry_px
                    actions.append(f"ENTRY {ticker}: {shares:.2f} shares @ ${entry_px:.2f} "
                                   f"(gap +{gap['gap_pct']}%)")

    # 4. Update NAV
    if state['positions']:
        pos_tickers = list(state['positions'].keys())
        prices = get_current_prices(pos_tickers)
        positions_value = sum(
            state['positions'][t]['shares'] * prices.get(t, state['positions'][t]['entry_price'])
            for t in pos_tickers
        )
    else:
        positions_value = 0

    state['nav'] = state['cash'] + positions_value

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"UPDATED STATE:")
    print(f"  NAV: ${state['nav']:.2f}  Cash: ${state['cash']:.2f}")
    print(f"  Positions: {len(state['positions'])}")
    print(f"  Record: {state['wins']}W-{state['losses']}L  Total PnL: ${state['total_pnl']:.2f}")
    if actions:
        print(f"  Actions taken:")
        for a in actions:
            print(f"    → {a}")
    print(f"  Beat history tracked: {len(state.get('beat_history', {}))} tickers")
    print(f"{'='*60}")

    if not check_only:
        save_state(state)
        save_trades(trades)
        print(f"\nState saved.")

    return state, actions


if __name__ == "__main__":
    check_only = "--check-only" in sys.argv
    run_engine(check_only=check_only)
