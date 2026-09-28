#!/usr/bin/env python3
"""
Volume Anomaly Paper Trading Engine — Multi-Day Surge (Variant E)
=================================================================
Validated: 5/5 gates, 4/5 adversarial
- Sharpe 2.37, WR 61.2%, MDD -5.0%, regime gap 0.338, perm p=0.004
- Buy sector ETFs when volume > 1.5x 20-day avg for 3 consecutive days
- Hold 10 trading days
- Works in both bull and bear regimes (institutional flow signal)

Usage: python3 volume_anomaly_paper_engine.py [--check-only]
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
STATE_FILE = Path("/home/jupiter/Lvl3Quant/data/paper_engines/volume_anomaly/state.json")
TRADES_FILE = Path("/home/jupiter/Lvl3Quant/data/paper_engines/volume_anomaly/trades.json")

UNIVERSE = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"
]

INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 3
HOLD_DAYS = 10  # Trading days
VOL_THRESHOLD = 1.5  # Volume must be > 1.5x 20-day avg
CONSECUTIVE_DAYS = 3  # Need 3 consecutive high-volume days
VOL_LOOKBACK = 20  # 20-day average volume


def load_state():
    """Load or initialize paper trading state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "nav": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "strategy": "volume_anomaly_multi_day_surge",
        "variant": "E",
        "backtest_sharpe": 2.367,
        "backtest_perm_p": 0.004,
        "adversarial_score": "4/5",
    }


def load_trades():
    """Load trade history."""
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def save_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def detect_volume_surges():
    """
    Detect ETFs with 3+ consecutive days of volume > 1.5x 20-day average.
    """
    signals = []
    end = datetime.now()
    start = end - timedelta(days=60)  # Extra buffer for 20-day avg

    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df is None or df.empty or len(df) < VOL_LOOKBACK + CONSECUTIVE_DAYS:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            # Calculate volume ratio
            df['vol_avg'] = df['Volume'].rolling(VOL_LOOKBACK).mean()
            df['vol_ratio'] = df['Volume'] / df['vol_avg']

            # Check last few days for consecutive high-volume
            recent = df.tail(CONSECUTIVE_DAYS + 2)  # Extra buffer

            # Check if last CONSECUTIVE_DAYS days all have vol_ratio > threshold
            last_n = df.tail(CONSECUTIVE_DAYS)
            if len(last_n) < CONSECUTIVE_DAYS:
                continue

            all_high = all(
                float(last_n['vol_ratio'].iloc[i]) >= VOL_THRESHOLD
                for i in range(CONSECUTIVE_DAYS)
            )

            if all_high:
                current_price = float(df['Close'].iloc[-1])
                avg_vol_ratio = float(last_n['vol_ratio'].mean())
                # Check if bullish (close > open on most recent day)
                bullish = float(df['Close'].iloc[-1]) >= float(df['Open'].iloc[-1])

                signals.append({
                    "ticker": ticker,
                    "signal_date": df.index[-1].strftime('%Y-%m-%d'),
                    "current_price": round(current_price, 2),
                    "avg_vol_ratio": round(avg_vol_ratio, 2),
                    "consecutive_days": CONSECUTIVE_DAYS,
                    "bullish": bullish,
                })
        except Exception as e:
            continue

    return signals


def get_current_prices(tickers):
    """Get current prices for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="2d", progress=False)
        if data is None or data.empty:
            return prices
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    prices[ticker] = float(data['Close'][ticker].iloc[-1])
                except (KeyError, IndexError):
                    pass
        else:
            prices[tickers[0]] = float(data['Close'].iloc[-1])
    except Exception:
        pass
    return prices


def run_engine(check_only=False):
    """Main paper trading engine loop."""
    state = load_state()
    trades = load_trades()
    actions = []

    print(f"{'='*60}")
    print(f"VOLUME ANOMALY — Paper Engine (Multi-Day Surge E)")
    print(f"{'='*60}")
    print(f"NAV: ${state['nav']:.2f}  Cash: ${state['cash']:.2f}")
    print(f"Positions: {len(state['positions'])} / {MAX_POSITIONS}")
    print(f"Record: {state['wins']}W-{state['losses']}L  PnL: ${state['total_pnl']:.2f}")
    print()

    # 1. Update existing positions
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
                exit_px = current_px * (1 - 2 / 10000)  # 0.02% slippage
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
                    "vol_ratio": pos.get('vol_ratio', 0),
                }
                trades.append(trade_record)
                actions.append(f"EXIT {ticker}: ${realized_pnl:+.2f} ({pnl_pct:+.1f}%)")

        for ticker in exits:
            del state['positions'][ticker]

    # 2. Scan for new volume surge entries
    open_slots = MAX_POSITIONS - len(state['positions'])

    if open_slots <= 0:
        print(f"\n⚠️  Max positions reached ({MAX_POSITIONS}) — no new entries")
    else:
        print(f"\nScanning for volume surges ({CONSECUTIVE_DAYS}d × {VOL_THRESHOLD}x)...")
        signals = detect_volume_surges()

        if not signals:
            print("  No qualifying volume surges found")
        else:
            print(f"  Found {len(signals)} signals:")
            new_entries = []

            for sig in sorted(signals, key=lambda x: x['avg_vol_ratio'], reverse=True):
                ticker = sig['ticker']

                if ticker in state['positions']:
                    print(f"    {ticker}: {sig['avg_vol_ratio']}x vol — ALREADY HOLDING")
                    continue

                print(f"    {ticker}: {sig['avg_vol_ratio']}x vol on {sig['signal_date']} "
                      f"{'(bullish)' if sig['bullish'] else '(bearish)'} — "
                      f"{'✅ ENTRY' if len(new_entries) < open_slots else '⏭ NO SLOT'}")

                if len(new_entries) < open_slots:
                    new_entries.append(sig)

            # Execute entries
            if new_entries and not check_only:
                alloc_per = min(
                    state['cash'] / len(new_entries),
                    state['cash'] / max(open_slots, 1)
                )

                for sig in new_entries:
                    ticker = sig['ticker']
                    entry_px = sig['current_price'] * (1 + 2 / 10000)  # slippage
                    shares = alloc_per / entry_px

                    state['positions'][ticker] = {
                        "entry_date": datetime.now().strftime('%Y-%m-%d'),
                        "entry_price": round(entry_px, 2),
                        "shares": round(shares, 4),
                        "signal_date": sig['signal_date'],
                        "vol_ratio": sig['avg_vol_ratio'],
                        "alloc_usd": round(alloc_per, 2),
                    }
                    state['cash'] -= shares * entry_px
                    actions.append(f"ENTRY {ticker}: {shares:.2f} shares @ ${entry_px:.2f} "
                                   f"(vol {sig['avg_vol_ratio']}x)")

    # 3. Update NAV
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

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"UPDATED STATE:")
    print(f"  NAV: ${state['nav']:.2f}  Cash: ${state['cash']:.2f}")
    print(f"  Positions: {len(state['positions'])}")
    print(f"  Record: {state['wins']}W-{state['losses']}L  Total PnL: ${state['total_pnl']:.2f}")
    if actions:
        print(f"  Actions taken:")
        for a in actions:
            print(f"    → {a}")
    print(f"{'='*60}")

    if not check_only:
        save_state(state)
        save_trades(trades)
        print(f"\nState saved.")

    return state, actions


if __name__ == "__main__":
    check_only = "--check-only" in sys.argv
    run_engine(check_only=check_only)
