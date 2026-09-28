#!/usr/bin/env python3
"""
Daily Earnings Surprise Signal Scanner
=======================================
Detects post-earnings gap-ups (>3%) as beat proxies, tracks consecutive beats,
generates buy signals with position sizing, and identifies sector sympathy plays.

Validated signal: buying after earnings beats, hold ~60 trading days.
Variant B (consecutive beaters) and Variant D (sector sympathy) are strongest.

Usage:
    python3 earnings_surprise_scanner.py          # normal run
    python3 earnings_surprise_scanner.py --lookback 5  # check last 5 days
"""

import json
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GROWTH_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER",
    "LYFT", "COIN", "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU",
]

SECTOR_MAP = {
    # Tech -> XLK
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "CRM": "XLK", "DDOG": "XLK", "TTD": "XLK", "NET": "XLK", "SHOP": "XLK",
    # Communications -> XLC
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "SNAP": "XLC",
    "PINS": "XLC", "RBLX": "XLC", "ROKU": "XLC",
    # Consumer Discretionary -> XLY
    "AMZN": "XLY", "TSLA": "XLY", "UBER": "XLY", "LYFT": "XLY",
    # Financials -> XLF
    "SOFI": "XLF", "HOOD": "XLF", "COIN": "XLF",
    # Other
    "PLTR": "PLTR",  # no clean sector ETF
}

GAP_THRESHOLD = 0.03          # 3% gap-up = earnings beat proxy
ACCOUNT_SIZE = 645.0          # $645 account
MAX_CONCURRENT = 3            # max 3 positions at once
HOLD_DAYS = 60                # ~60 trading days hold
VIX_KILL = 20.0               # VIX threshold
LOOKBACK_DAYS = 5             # how many days of price data to fetch for gap detection

BEAT_HISTORY_PATH = Path("/home/jupiter/Lvl3Quant/data/earnings_beat_history.json")
SIGNAL_OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/state/earnings_surprise_signals.json")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_beat_history() -> dict:
    """Load or initialize the consecutive-beat tracking file."""
    if BEAT_HISTORY_PATH.exists():
        try:
            with open(BEAT_HISTORY_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_beat_history(history: dict):
    BEAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEAT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def save_signal_output(output: dict):
    SIGNAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)


def business_days_ahead(start_date, n_days):
    """Estimate the date n business days from start_date."""
    current = start_date
    added = 0
    while added < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


def confidence_label(consecutive_beats: int) -> str:
    if consecutive_beats >= 3:
        return "HIGH"
    elif consecutive_beats == 2:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_price_data(tickers: list, period: str = "1mo") -> dict:
    """Fetch recent OHLCV for a list of tickers. Returns dict of DataFrames."""
    results = {}
    # Batch download for efficiency
    try:
        data = yf.download(tickers, period=period, group_by="ticker",
                           progress=False, threads=True)
        if data is not None and not data.empty:
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        df = data
                    else:
                        df = data[ticker].dropna(how="all")
                    if df is not None and not df.empty:
                        results[ticker] = df
                except (KeyError, TypeError):
                    pass
    except Exception as e:
        print(f"  [WARN] Batch download failed: {e}. Falling back to individual downloads.")

    # Fill in any missing tickers individually
    for ticker in tickers:
        if ticker not in results:
            try:
                t = yf.Ticker(ticker)
                df = t.history(period=period)
                if df is not None and not df.empty:
                    results[ticker] = df
            except Exception:
                print(f"  [WARN] Could not fetch data for {ticker}")

    return results


def fetch_market_data() -> dict:
    """Fetch SPY and VIX data for kill switch evaluation."""
    result = {}
    for sym in ["SPY", "^VIX"]:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo")
            if df is not None and not df.empty:
                result[sym] = df
        except Exception as e:
            print(f"  [WARN] Could not fetch {sym}: {e}")
    return result


# ---------------------------------------------------------------------------
# DETECTION LOGIC
# ---------------------------------------------------------------------------

def detect_gaps(price_data: dict, lookback: int = 2) -> list:
    """
    For each stock, check the last `lookback` trading days for a >3% gap-up
    from prior close to open (earnings beat proxy).

    Returns list of dicts with gap details.
    """
    gaps = []
    today = datetime.now().date()

    for ticker, df in price_data.items():
        if len(df) < 3:
            continue

        # Check last `lookback` days
        for i in range(-lookback, 0):
            try:
                row = df.iloc[i]
                prev_row = df.iloc[i - 1]

                open_price = float(row["Open"])
                prev_close = float(prev_row["Close"])
                current_close = float(row["Close"])

                if prev_close <= 0:
                    continue

                gap_pct = (open_price - prev_close) / prev_close

                if gap_pct >= GAP_THRESHOLD:
                    # Get the date of this gap
                    gap_date = df.index[i]
                    if hasattr(gap_date, 'date'):
                        gap_date = gap_date.date()

                    # Only report gaps from the last 2 calendar days
                    if (today - gap_date).days > 3:
                        continue

                    gaps.append({
                        "ticker": ticker,
                        "gap_pct": round(gap_pct * 100, 2),
                        "gap_date": str(gap_date),
                        "open": round(open_price, 2),
                        "prev_close": round(prev_close, 2),
                        "current_price": round(current_close, 2),
                        "sector_etf": SECTOR_MAP.get(ticker, "N/A"),
                    })
            except (IndexError, KeyError, TypeError):
                continue

    return gaps


def check_kill_switch(market_data: dict) -> tuple:
    """
    Returns (is_active: bool, reason: str).
    Kill switch active if VIX > 20 AND SPY < 50-day SMA.
    """
    spy_df = market_data.get("SPY")
    vix_df = market_data.get("^VIX")

    if spy_df is None or vix_df is None:
        return False, "Could not fetch market data — proceeding with caution"

    try:
        current_vix = float(vix_df["Close"].iloc[-1])
        current_spy = float(spy_df["Close"].iloc[-1])
        spy_sma50 = float(spy_df["Close"].rolling(50).mean().iloc[-1])

        vix_high = current_vix > VIX_KILL
        spy_below_sma = current_spy < spy_sma50

        if vix_high and spy_below_sma:
            return True, (
                f"VIX={current_vix:.1f} (>{VIX_KILL}) AND "
                f"SPY={current_spy:.2f} < 50-SMA={spy_sma50:.2f}"
            )
        else:
            return False, (
                f"VIX={current_vix:.1f}, SPY={current_spy:.2f}, "
                f"50-SMA={spy_sma50:.2f} — conditions OK"
            )
    except Exception as e:
        return False, f"Kill switch check error: {e} — proceeding with caution"


def update_beat_history(gaps: list, history: dict) -> dict:
    """Update consecutive beat tracking. Returns updated history."""
    for gap in gaps:
        ticker = gap["ticker"]
        gap_date = gap["gap_date"]

        if ticker not in history:
            history[ticker] = {
                "consecutive_beats": 1,
                "beat_dates": [gap_date],
                "last_beat": gap_date,
            }
        else:
            rec = history[ticker]
            # Don't double-count same date
            if gap_date not in rec.get("beat_dates", []):
                rec["consecutive_beats"] = rec.get("consecutive_beats", 0) + 1
                rec.setdefault("beat_dates", []).append(gap_date)
                rec["last_beat"] = gap_date

    return history


def generate_signals(gaps: list, history: dict, kill_active: bool) -> list:
    """Generate buy signals with position sizing."""
    if kill_active or not gaps:
        return []

    signals = []
    position_size = ACCOUNT_SIZE / MAX_CONCURRENT  # equal weight

    for gap in gaps:
        ticker = gap["ticker"]
        price = gap["current_price"]
        consec = history.get(ticker, {}).get("consecutive_beats", 1)
        conf = confidence_label(consec)

        shares = int(position_size / price) if price > 0 else 0
        if shares < 1:
            shares = 1  # minimum 1 share

        cost = round(shares * price, 2)
        exit_date = business_days_ahead(datetime.now().date(), HOLD_DAYS)

        signals.append({
            "ticker": ticker,
            "action": "BUY",
            "gap_pct": gap["gap_pct"],
            "gap_date": gap["gap_date"],
            "current_price": price,
            "consecutive_beats": consec,
            "confidence": conf,
            "shares": shares,
            "position_cost": cost,
            "sector_etf": gap["sector_etf"],
            "sympathy_play": f"Also consider {gap['sector_etf']}" if gap["sector_etf"] not in ("N/A", ticker) else "No sector ETF play",
            "hold_days": HOLD_DAYS,
            "estimated_exit": str(exit_date),
        })

    # Sort by confidence (HIGH first), then by gap size
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    signals.sort(key=lambda s: (conf_order.get(s["confidence"], 3), -s["gap_pct"]))

    return signals


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_summary(signals: list, gaps: list, kill_active: bool, kill_reason: str,
                  history: dict):
    """Print clean human-readable summary."""
    print("=" * 70)
    print(f"  EARNINGS SURPRISE SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Kill switch status
    if kill_active:
        print(f"\n  *** KILL SWITCH ACTIVE — NO ENTRIES ***")
        print(f"  Reason: {kill_reason}")
        print("=" * 70)
        return
    else:
        print(f"\n  Market: {kill_reason}")

    # Gaps detected
    print(f"\n  Gaps detected (>{GAP_THRESHOLD*100:.0f}% gap-up): {len(gaps)}")
    if not gaps:
        print("  No earnings gap-ups detected in the last 2 trading days.")
        print("=" * 70)
        return

    for g in gaps:
        print(f"    {g['ticker']:6s}  +{g['gap_pct']:.1f}%  on {g['gap_date']}  "
              f"(open {g['open']} vs prev close {g['prev_close']})")

    # Signals
    print(f"\n  {'─' * 66}")
    print(f"  SIGNALS ({len(signals)}):")
    print(f"  {'─' * 66}")

    if not signals:
        print("  No actionable signals.")
    else:
        for s in signals:
            print(f"\n  {s['ticker']} — {s['confidence']} confidence")
            print(f"    Gap: +{s['gap_pct']:.1f}% on {s['gap_date']}")
            print(f"    Consecutive beats: {s['consecutive_beats']}")
            print(f"    Action: BUY {s['shares']} shares @ ${s['current_price']:.2f} "
                  f"(${s['position_cost']:.2f})")
            print(f"    Sector ETF: {s['sector_etf']}  |  {s['sympathy_play']}")
            print(f"    Hold: {s['hold_days']} trading days → exit ~{s['estimated_exit']}")

    # Beat history summary
    tracked = {k: v for k, v in history.items() if v.get("consecutive_beats", 0) >= 2}
    if tracked:
        print(f"\n  {'─' * 66}")
        print(f"  CONSECUTIVE BEATERS (variant B — strongest signal):")
        print(f"  {'─' * 66}")
        for ticker, rec in sorted(tracked.items(),
                                   key=lambda x: -x[1].get("consecutive_beats", 0)):
            print(f"    {ticker:6s}  {rec['consecutive_beats']} consecutive beats  "
                  f"(last: {rec.get('last_beat', 'N/A')})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily Earnings Surprise Scanner")
    parser.add_argument("--lookback", type=int, default=2,
                        help="How many trading days back to check for gaps (default: 2)")
    args = parser.parse_args()

    print("Fetching price data for growth universe...")
    price_data = fetch_price_data(GROWTH_UNIVERSE, period="1mo")
    print(f"  Got data for {len(price_data)}/{len(GROWTH_UNIVERSE)} stocks")

    print("Fetching market data (SPY, VIX)...")
    market_data = fetch_market_data()

    # Kill switch check
    kill_active, kill_reason = check_kill_switch(market_data)

    # Detect gaps
    gaps = detect_gaps(price_data, lookback=args.lookback)

    # Load and update beat history
    history = load_beat_history()
    if gaps:
        history = update_beat_history(gaps, history)
        save_beat_history(history)
        print(f"  Updated beat history ({len(gaps)} new gap(s) detected)")

    # Generate signals
    signals = generate_signals(gaps, history, kill_active)

    # Print summary
    print_summary(signals, gaps, kill_active, kill_reason, history)

    # Save state
    output = {
        "timestamp": datetime.now().isoformat(),
        "kill_switch_active": kill_active,
        "kill_switch_reason": kill_reason,
        "gaps_detected": gaps,
        "signals": signals,
        "beat_history": history,
        "config": {
            "gap_threshold": GAP_THRESHOLD,
            "account_size": ACCOUNT_SIZE,
            "max_concurrent": MAX_CONCURRENT,
            "hold_days": HOLD_DAYS,
            "vix_kill_threshold": VIX_KILL,
        },
    }
    save_signal_output(output)
    print(f"\nState saved. Scanner complete.")

    return 0 if not kill_active else 1


if __name__ == "__main__":
    sys.exit(main())
