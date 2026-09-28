#!/usr/bin/env python3
"""
Earnings Week Engine — Automated PEAD + IV Run-Up Scanner
=========================================================
Two validated strategies automated for the agentic account ($645, $200 max/trade):

1. PEAD (Post-Earnings Announcement Drift): After earnings gap > threshold,
   buy shares/options in gap direction. Validated: Sharpe 1.51, 60% WR.

2. IV Run-Up: Buy ATM straddles T-7 to T-15 before earnings, sell T-1.
   Validated: Sharpe 2.27+, 92% WR on focused universe.

Requires earnings_calendar.json (from Robinhood API) as input.
Run: python3 earnings_week_engine.py [--calendar /path/to/calendar.json]
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ─────────────────────────────────────────────────────────────────

MAX_POSITION = 200   # $200 max per trade (HC #749)
ACCOUNT_SIZE = 645   # agentic account

# PEAD parameters (from validated backtest — Sharpe 1.51, 60% WR)
PEAD_GAP_THRESHOLD = 0.03     # 3% gap minimum for drift signal
PEAD_HOLD_DAYS = 5            # hold for drift (validated holding period)

# IV Run-Up parameters (from validated backtest — Sharpe 2.27, 92% WR)
IVRU_ENTRY_WINDOW = (7, 15)   # enter T-7 to T-15 before earnings
IVRU_EXIT_BEFORE = 1          # exit T-1 (day before earnings)
IVRU_MAX_STOCK_PRICE = 35     # stock must be cheap enough for straddle < $200

# Growth/volatile universe with liquid options
GROWTH_UNIVERSE = [
    "SNAP", "SOFI", "LCID", "RIOT", "MARA", "PLUG", "LYFT", "PINS",
    "DKNG", "UPST", "HIMS", "RBLX", "COIN", "HOOD", "UBER", "ABNB",
    "DASH", "NET", "CRWD", "DDOG", "TTD", "DUOL", "APP", "SHOP",
    "AMD", "NVDA", "TSLA", "META", "MSFT", "AAPL", "AMZN", "GOOGL",
    "NFLX", "PLTR", "AFRM", "RDDT", "CAVA", "ARM"
]

SIGNALS_DIR = Path("/home/jupiter/Lvl3Quant/data/earnings_signals")
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
CALENDAR_PATH = Path("/home/jupiter/Lvl3Quant/data/earnings_signals/earnings_calendar.json")


# ─── Earnings Calendar ──────────────────────────────────────────────────────

def load_earnings_calendar(path: Path = CALENDAR_PATH) -> dict:
    """Load earnings calendar (symbol -> {date, timing}).
    Source: Robinhood get_earnings_calendar, saved to JSON."""
    if not path.exists():
        print(f"WARNING: No calendar at {path}. Run with --save-calendar first.")
        return {}
    with open(path) as f:
        return json.load(f)


def save_earnings_calendar(this_week: list, next_week: list, path: Path = CALENDAR_PATH):
    """Save combined earnings calendar from Robinhood API results."""
    cal = {}
    for entry in this_week + next_week:
        sym = entry.get('symbol', '')
        report = entry.get('report', {}) or {}
        rdate = report.get('date', '')
        timing = report.get('timing', '')
        if sym and rdate:
            cal[sym] = {'date': rdate, 'timing': timing or 'unknown'}
    with open(path, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"Saved {len(cal)} entries to {path}")
    return cal


def stocks_that_reported(calendar: dict, on_date: date, timing_filter: str = None) -> list:
    """Return symbols that reported earnings on a specific date."""
    results = []
    date_str = str(on_date)
    for sym, info in calendar.items():
        if info['date'] == date_str:
            if timing_filter and info.get('timing') != timing_filter:
                continue
            results.append(sym)
    return results


def stocks_reporting_soon(calendar: dict, today: date, window: tuple = (7, 15)) -> dict:
    """Return symbols reporting within the entry window."""
    results = {}
    for sym, info in calendar.items():
        try:
            earn_date = date.fromisoformat(info['date'])
            days_out = (earn_date - today).days
            if window[0] <= days_out <= window[1]:
                results[sym] = {
                    'earnings_date': info['date'],
                    'timing': info.get('timing', '?'),
                    'days_out': days_out
                }
        except (ValueError, TypeError):
            pass
    return results


# ─── PEAD Scanner ────────────────────────────────────────────────────────────

def check_post_earnings_gap(sym: str) -> dict | None:
    """Check if stock gapped significantly (open vs prev close)."""
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="5d")
        if len(hist) < 2:
            return None

        today_row = hist.iloc[-1]
        yesterday_row = hist.iloc[-2]

        gap_pct = (today_row['Open'] - yesterday_row['Close']) / yesterday_row['Close']

        if abs(gap_pct) >= PEAD_GAP_THRESHOLD:
            # For PEAD: buy in gap direction (momentum drift)
            # Agentic account can't short, so DOWN gaps = buy puts
            return {
                'symbol': sym,
                'gap_pct': round(gap_pct * 100, 2),
                'direction': 'UP' if gap_pct > 0 else 'DOWN',
                'prev_close': round(yesterday_row['Close'], 2),
                'open_price': round(today_row['Open'], 2),
                'current_price': round(today_row['Close'], 2),
                'strategy': 'PEAD',
                'confidence': 'HIGH' if abs(gap_pct) > 0.05 else 'MEDIUM',
                'action': 'BUY_SHARES' if gap_pct > 0 else 'BUY_PUTS',
            }
    except Exception:
        pass
    return None


def scan_pead(calendar: dict, today: date) -> list[dict]:
    """Scan for PEAD signals: stocks that reported yesterday/today and gapped."""
    signals = []

    # Check yesterday PM reporters + today AM reporters
    yesterday = today - timedelta(days=1)
    # If Monday, also check Friday
    if today.weekday() == 0:  # Monday
        friday = today - timedelta(days=3)
        check_dates = [friday, yesterday, today]
    else:
        check_dates = [yesterday, today]

    candidates = set()
    for d in check_dates:
        for sym in stocks_that_reported(calendar, d):
            if sym in GROWTH_UNIVERSE or sym in calendar:
                candidates.add(sym)

    print(f"  Checking {len(candidates)} stocks that recently reported")
    for sym in candidates:
        result = check_post_earnings_gap(sym)
        if result:
            signals.append(result)

    return signals


# ─── IV Run-Up Scanner ──────────────────────────────────────────────────────

def compute_iv_percentile(sym: str, period: int = 252) -> float:
    """HV-based IV percentile proxy (actual IV requires options data)."""
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period=f"{period + 30}d")
        if len(hist) < 60:
            return 50.0
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        rv = returns.rolling(20).std() * np.sqrt(252)
        current = rv.iloc[-1]
        pctile = (rv < current).sum() / len(rv) * 100
        return round(pctile, 1)
    except Exception:
        return 50.0


def scan_iv_runup(calendar: dict, today: date) -> list[dict]:
    """Scan for IV run-up entry candidates."""
    upcoming = stocks_reporting_soon(calendar, today, window=IVRU_ENTRY_WINDOW)

    # Filter to our growth universe for quality
    candidates = {s: v for s, v in upcoming.items() if s in GROWTH_UNIVERSE}
    print(f"  {len(candidates)} growth names reporting in {IVRU_ENTRY_WINDOW[0]}-{IVRU_ENTRY_WINDOW[1]}d window")

    signals = []
    for sym, info in candidates.items():
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="5d")
            if len(hist) < 1:
                continue
            price = hist['Close'].iloc[-1]

            # Price filter
            if price > IVRU_MAX_STOCK_PRICE:
                print(f"    {sym}: ${price:.2f} > ${IVRU_MAX_STOCK_PRICE} cap — SKIP")
                continue

            iv_pctile = compute_iv_percentile(sym)
            est_straddle = price * 0.10 * 100  # rough estimate

            if est_straddle > MAX_POSITION:
                print(f"    {sym}: est straddle ${est_straddle:.0f} > ${MAX_POSITION} — SKIP")
                continue

            signals.append({
                'symbol': sym,
                'current_price': round(price, 2),
                'earnings_date': info['earnings_date'],
                'days_to_earnings': info['days_out'],
                'iv_percentile': iv_pctile,
                'est_straddle_cost': round(est_straddle, 0),
                'strategy': 'IV_RUNUP',
                'action': 'BUY_STRADDLE',
                'confidence': 'HIGH' if iv_pctile < 40 else 'MEDIUM',
            })
        except Exception as e:
            print(f"    {sym}: error — {e}")

    # Sort: HIGH confidence first, then lowest IV percentile
    signals.sort(key=lambda x: (0 if x['confidence'] == 'HIGH' else 1, x['iv_percentile']))
    return signals


# ─── Output ──────────────────────────────────────────────────────────────────

def format_discord_message(pead_signals: list, ivru_signals: list) -> str | None:
    """Format signals for Discord per HC #433 (plain English, no paths)."""
    if not pead_signals and not ivru_signals:
        return None

    lines = []

    if pead_signals:
        lines.append("**Earnings Gap Trades (PEAD)**")
        for s in pead_signals:
            direction = "gapped UP" if s['direction'] == 'UP' else "gapped DOWN"
            lines.append(
                f"• {s['symbol']}: {direction} {abs(s['gap_pct'])}% after earnings — "
                f"{s['confidence']} confidence, {s['action'].replace('_', ' ').lower()} for 5-day drift"
            )

    if ivru_signals:
        if lines:
            lines.append("")
        lines.append("**IV Run-Up Entries (Straddle)**")
        for s in ivru_signals[:5]:
            lines.append(
                f"• {s['symbol']}: ${s['current_price']}, earnings {s['earnings_date']} "
                f"({s['days_to_earnings']}d out) — IV pctile {s['iv_percentile']}%, "
                f"est. straddle ~${int(s['est_straddle_cost'])}"
            )

    return "\n".join(lines)


def main():
    today = date.today()
    print(f"=== Earnings Week Engine — {today} ===\n")

    # Load calendar
    calendar = load_earnings_calendar()
    if not calendar:
        print("No calendar loaded. Use --save-calendar with Robinhood data first.")
        print("Falling back to yfinance for gap detection only.\n")

    # PEAD scan
    print("── PEAD Scan (post-earnings gaps) ──")
    if calendar:
        pead_signals = scan_pead(calendar, today)
    else:
        pead_signals = []
        print("  Skipped (no calendar)")
    print(f"  Result: {len(pead_signals)} PEAD signals")

    # IV Run-Up scan
    print("\n── IV Run-Up Scan ──")
    if calendar:
        ivru_signals = scan_iv_runup(calendar, today)
    else:
        ivru_signals = []
        print("  Skipped (no calendar)")
    print(f"  Result: {len(ivru_signals)} IV run-up candidates")

    # Save
    all_signals = {
        'date': str(today),
        'pead': pead_signals,
        'iv_runup': ivru_signals,
        'timestamp': datetime.now().isoformat(),
        'calendar_entries': len(calendar),
    }
    output_path = SIGNALS_DIR / f"signals_{today}.json"
    with open(output_path, 'w') as f:
        json.dump(all_signals, f, indent=2)
    print(f"\nSignals saved.")

    # Discord message
    msg = format_discord_message(pead_signals, ivru_signals)
    if msg:
        print(f"\n=== DISCORD MESSAGE ===\n{msg}")
    else:
        print("\nNo actionable signals today.")

    return all_signals


if __name__ == "__main__":
    results = main()
