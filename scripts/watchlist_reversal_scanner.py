#!/usr/bin/env python3
"""
Watchlist Reversal Scanner — detects oversold bounce setups on watched tickers.

Generalizes the META reversal watcher (Session 87) into a configurable scanner
that captures individual stock reversal signals and routes them to:
  1. Discord alert (via alert file → autonomy_inject)
  2. Morning execution queue (if after hours)
  3. Signal aggregator pending list (if during market hours)

Built Session 88: user said "fix the pipeline to capture those as well"
after META reversal alert fired but wasn't captured for action.

Watchlist is configurable via state/reversal_watchlist.json.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("yfinance not available")
    sys.exit(0)

import warnings
warnings.filterwarnings('ignore')

BASE = Path("/home/jupiter/Lvl3Quant")
WATCHLIST_FILE = BASE / "state" / "reversal_watchlist.json"
STATE_FILE = BASE / "state" / "watchlist_reversal_state.json"
ALERT_FILE = BASE / "state" / "signal_flip_alert.txt"  # Reuse flip alert mechanism
MORNING_QUEUE = BASE / "state" / "morning_execution_queue.json"
PENDING_FILE = BASE / "state" / "pending_entries.json"

# Default watchlist — user can add/remove via state file
DEFAULT_WATCHLIST = {
    "tickers": {
        "META": {
            "bounce_level": 530,
            "early_warning": 540,
            "note": "User watching for personal account call at $530 bounce zone (80% historical bounce rate)",
            "account": "personal"
        }
    },
    "settings": {
        "rsi_oversold": 35,
        "rsi_turning_up_range": [25, 50],
        "volume_surge_threshold": 1.3,
        "big_green_threshold": 0.015,
        "min_selloff_5d": -0.03
    }
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def is_market_open():
    """Check if market is in regular trading hours."""
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        from market_status import get_market_status
        status = get_market_status()
        return status.get("is_rth", False) or status.get("session", "") == "regular"
    except Exception:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        return now_et.weekday() < 5 and 9 <= now_et.hour < 16


def compute_rsi(close_prices, period=14):
    """Compute RSI from a list of close prices."""
    if len(close_prices) < period + 1:
        return 50.0
    deltas = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
    recent = deltas[-period:]
    gains = [max(d, 0) for d in recent]
    losses = [abs(min(d, 0)) for d in recent]
    avg_gain = sum(gains) / period
    avg_loss = max(sum(losses) / period, 0.001)
    return 100 - (100 / (1 + avg_gain / avg_loss))


def scan_ticker(ticker, config, settings):
    """Scan a single ticker for reversal signals."""
    try:
        data = yf.download(ticker, period="60d", interval="1d", progress=False)
        if data.empty or len(data) < 20:
            return None

        # Handle multi-index columns
        if hasattr(data.columns, 'levels'):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        close = [float(x) for x in data['Close'].values]
        volume = [float(x) for x in data['Volume'].values]
        current = close[-1]
        prev = close[-2]

        # RSI calculations
        rsi14 = compute_rsi(close, 14)
        rsi5 = compute_rsi(close, 5)

        # Volume analysis
        avg_vol_20 = sum(volume[-21:-1]) / 20 if len(volume) >= 21 else sum(volume) / len(volume)
        vol_ratio = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # Price action
        green = current > prev
        pct_change = (current - prev) / prev
        big_green = pct_change > settings.get("big_green_threshold", 0.015)
        vol_surge = vol_ratio > settings.get("volume_surge_threshold", 1.3)

        # RSI turning up from oversold
        rsi_range = settings.get("rsi_turning_up_range", [25, 50])
        rsi_turning = rsi_range[0] < rsi14 < rsi_range[1] and green

        # 5-day selloff check (signal is more valuable after a selloff)
        if len(close) >= 6:
            ret_5d = (close[-1] / close[-6]) - 1
        else:
            ret_5d = 0

        # Price level checks (custom per ticker)
        bounce_level = config.get("bounce_level")
        early_warning = config.get("early_warning")
        hit_bounce = current <= bounce_level if bounce_level else False
        near_bounce = current <= early_warning if early_warning else False

        # Determine alert level
        alert = False
        alert_type = None
        alert_msg = ""

        if hit_bounce:
            alert = True
            alert_type = "BOUNCE_LEVEL"
            alert_msg = (f"{ticker} HIT ${bounce_level} support! Price ${current:.2f}. "
                        f"{config.get('note', 'Consider entry.')}")
        elif big_green and vol_surge:
            alert = True
            alert_type = "BIG_GREEN_VOLUME"
            alert_msg = (f"{ticker} big green reversal: ${current:.2f} (+{pct_change*100:.1f}%) "
                        f"on {vol_ratio:.1f}x avg volume. RSI {rsi14:.0f}.")
        elif rsi_turning and ret_5d < settings.get("min_selloff_5d", -0.03):
            alert = True
            alert_type = "RSI_REVERSAL"
            alert_msg = (f"{ticker} RSI reversal: ${current:.2f}, RSI {rsi14:.0f} turning up "
                        f"after {ret_5d*100:.1f}% selloff.")
        elif near_bounce:
            alert = True
            alert_type = "APPROACHING"
            alert_msg = (f"{ticker} approaching ${bounce_level} support "
                        f"(now ${current:.2f}, {((current/bounce_level)-1)*100:.1f}% above)")

        result = {
            "ticker": ticker,
            "price": round(current, 2),
            "change_pct": round(pct_change * 100, 2),
            "rsi14": round(rsi14, 1),
            "rsi5": round(rsi5, 1),
            "vol_ratio": round(vol_ratio, 2),
            "ret_5d": round(ret_5d * 100, 2),
            "green_day": green,
            "big_green": big_green,
            "vol_surge": vol_surge,
            "rsi_turning": rsi_turning,
            "hit_bounce": hit_bounce,
            "near_bounce": near_bounce,
            "alert": alert,
            "alert_type": alert_type,
            "alert_msg": alert_msg,
            "account": config.get("account", "agentic"),
            "timestamp": datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"[{datetime.now()}] Error scanning {ticker}: {e}")
        return None


def route_alert(result, market_open):
    """Route an alert to the appropriate destination."""
    ticker = result["ticker"]
    msg = result["alert_msg"]
    account = result.get("account", "agentic")

    # Always write to alert file for autonomy_inject pickup
    try:
        # Append to existing alerts (multiple tickers may fire)
        existing = ""
        if Path(ALERT_FILE).exists():
            existing = Path(ALERT_FILE).read_text() + "\n"
        with open(ALERT_FILE, 'w') as f:
            f.write(existing + msg)
    except OSError:
        pass

    if account == "personal":
        # Personal account alerts: just notify via inject, user decides
        print(f"  → PERSONAL ACCOUNT ALERT: {msg}")
        return

    # Agentic account: route to execution pipeline
    if market_open:
        # During hours: add to pending entries for next execution window
        try:
            pending = load_json(PENDING_FILE)
            if "pending" not in pending:
                pending["pending"] = []

            # Don't duplicate
            existing_tickers = {p.get("ticker") for p in pending["pending"]}
            if ticker not in existing_tickers:
                pending["pending"].append({
                    "ticker": ticker,
                    "direction": "bull",  # Reversal = buy the bounce
                    "confidence": 0.70,
                    "n_sources": 1,
                    "sources": [f"reversal_scanner({result['alert_type']})"],
                    "rationale": msg,
                    "deferred_reason": "Reversal signal — needs aggregator confirmation",
                    "valid_until": datetime.now().strftime("%Y-%m-%dT19:00:00"),
                    "source": "watchlist_reversal_scanner",
                    "exit_guidance": {
                        "take_profit_pct": 0.30,
                        "stop_loss_pct": 0.25,
                        "max_hold_days": 5,
                        "trailing_stop": 0.50
                    }
                })
                save_json(PENDING_FILE, pending)
                print(f"  → Added to pending entries for next execution window")
        except Exception as e:
            print(f"  → Error adding to pending: {e}")
    else:
        # After hours: queue for morning
        try:
            queue = load_json(MORNING_QUEUE)
            if not isinstance(queue, dict):
                queue = {}
            if "flips" not in queue:
                queue["flips"] = []
                queue["status"] = "PENDING"

            queue["flips"].append({
                "ticker": ticker,
                "direction": "bull",
                "confidence": 0.70,
                "n_confirming": 1,
                "n_prior": 0,
                "new_sources": [f"reversal_scanner({result['alert_type']})"],
                "source": "watchlist_reversal_scanner"
            })
            queue["queued_at"] = datetime.now().isoformat()
            queue["status"] = "PENDING"
            save_json(MORNING_QUEUE, queue)
            print(f"  → Queued for morning execution")
        except Exception as e:
            print(f"  → Error queueing for morning: {e}")


def main():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Watchlist reversal scanner running...")

    # Load or create watchlist
    watchlist = load_json(WATCHLIST_FILE)
    if not watchlist or "tickers" not in watchlist:
        watchlist = DEFAULT_WATCHLIST
        save_json(WATCHLIST_FILE, watchlist)
        print(f"  Created default watchlist with {len(watchlist['tickers'])} tickers")

    settings = watchlist.get("settings", DEFAULT_WATCHLIST["settings"])
    tickers = watchlist.get("tickers", {})

    if not tickers:
        print(f"  No tickers in watchlist — skipping")
        return

    market_open = is_market_open()
    state = load_json(STATE_FILE)
    alerts_fired = 0

    for ticker, config in tickers.items():
        result = scan_ticker(ticker, config, settings)
        if result is None:
            continue

        # Update state
        state[ticker] = result
        print(f"  {ticker}: ${result['price']:.2f} ({result['change_pct']:+.1f}%), "
              f"RSI(14)={result['rsi14']:.0f}, vol={result['vol_ratio']:.1f}x "
              f"{'⚡ ALERT: ' + result['alert_type'] if result['alert'] else 'no signal'}")

        if result["alert"]:
            route_alert(result, market_open)
            alerts_fired += 1

    # Save state
    state["last_scan"] = timestamp
    state["n_alerts"] = alerts_fired
    save_json(STATE_FILE, state)

    if alerts_fired:
        print(f"\n  {alerts_fired} alert(s) fired — routed to "
              f"{'execution pipeline' if market_open else 'morning queue'}")
    else:
        print(f"  No alerts. All tickers quiet.")


if __name__ == "__main__":
    main()
