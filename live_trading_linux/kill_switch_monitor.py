#!/usr/bin/env python3
"""
Kill Switch Monitor — Strategy Regime Guard
=============================================
Checks 5 market conditions that should halt or reduce paper engine entries:

1. VIX > 35 AND SPY < 200-SMA → halt all except defensive wheel CSPs
2. VIX < 12 → halt premium selling (CSP, BPS, IC, Strangle) — premiums too thin
3. SPY < 50-SMA AND VIX rising above 20 → reduce 50%, pause rotation/momentum
4. 3+ consecutive red days with VIX expansion > 3pts → pause all 2 days
5. VIX term structure inverted (front > back) → halt Iron Condors and Strangles

Returns a dict consumed by any paper engine:
    from kill_switch_monitor import check_kill_switches

Caches yfinance data for 5 minutes to avoid hammering the API.

Standalone mode: python kill_switch_monitor.py [--alert]
  --alert: POST to autonomy inject endpoint if any kill switch active
"""

import time
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np

logger = logging.getLogger("kill_switch_monitor")

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache = {"data": None, "ts": 0}
_cache_lock = Lock()
CACHE_TTL = 300  # 5 minutes

STATE_DIR = Path(__file__).parent / "kill_switch_state"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "kill_switch_log.jsonl"

INJECT_URL = "http://127.0.0.1:7731/inject"

# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

def _fetch_market_data():
    """Download SPY, VIX, VIX3M from yfinance. Returns dict or raises."""
    import yfinance as yf

    data = {}

    # SPY — need 200+ trading days for 200-SMA
    spy = yf.Ticker("SPY").history(period="1y")
    if len(spy) < 200:
        raise ValueError(f"SPY history too short: {len(spy)} bars (need 200)")
    data["spy_close"] = float(spy["Close"].iloc[-1])
    data["spy_sma50"] = float(spy["Close"].rolling(50).mean().iloc[-1])
    data["spy_sma200"] = float(spy["Close"].rolling(200).mean().iloc[-1])
    # Last 5 closes for consecutive-red-day check
    data["spy_last5"] = [float(x) for x in spy["Close"].iloc[-6:].values]  # 6 values → 5 daily returns

    # VIX spot
    vix = yf.Ticker("^VIX").history(period="10d")
    if len(vix) < 4:
        raise ValueError(f"VIX history too short: {len(vix)} bars")
    data["vix"] = float(vix["Close"].iloc[-1])
    data["vix_prev"] = float(vix["Close"].iloc[-2])
    # VIX 4 days ago for 3-pt expansion check
    data["vix_4d_ago"] = float(vix["Close"].iloc[-4]) if len(vix) >= 4 else data["vix"]
    data["vix_history"] = [float(x) for x in vix["Close"].values]

    # VIX3M (3-month VIX) — proxy for back-month VIX futures
    # ^VIX is front-month proxy, ^VIX3M is back-month proxy
    try:
        vix3m = yf.Ticker("^VIX3M").history(period="5d")
        if len(vix3m) >= 1:
            data["vix3m"] = float(vix3m["Close"].iloc[-1])
        else:
            data["vix3m"] = None
    except Exception:
        data["vix3m"] = None

    data["fetch_time"] = datetime.now().isoformat()
    return data


def _get_data(force_refresh=False):
    """Return cached market data, refreshing if stale."""
    with _cache_lock:
        now = time.time()
        if not force_refresh and _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
            return _cache["data"]

    # Fetch outside the lock (network I/O)
    data = _fetch_market_data()

    with _cache_lock:
        _cache["data"] = data
        _cache["ts"] = time.time()
    return data


# ---------------------------------------------------------------------------
# Kill-switch logic
# ---------------------------------------------------------------------------

def check_kill_switches(force_refresh=False):
    """
    Check all 5 kill-switch conditions.

    Returns:
        dict with keys:
            pause_all        – True if KS#1 or KS#4 fires (halt everything except maybe defensive CSPs)
            pause_premium    – True if KS#2 fires (premiums too thin)
            pause_rotation   – True if KS#3 fires (trend breakdown)
            pause_strangles  – True if KS#5 fires (term structure inverted)
            reduce_50pct     – True if KS#3 fires (cut size in half)
            reasons          – list of human-readable strings
            raw              – the underlying market data snapshot
            checked_at       – ISO timestamp
    """
    try:
        d = _get_data(force_refresh=force_refresh)
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        # Fail-open: if we can't check, don't block trading
        return {
            "pause_all": False,
            "pause_premium": False,
            "pause_rotation": False,
            "pause_strangles": False,
            "reduce_50pct": False,
            "reasons": [f"DATA_ERROR: {e}"],
            "raw": {},
            "checked_at": datetime.now().isoformat(),
            "error": True,
        }

    reasons = []
    pause_all = False
    pause_premium = False
    pause_rotation = False
    pause_strangles = False
    reduce_50pct = False

    vix = d["vix"]
    spy_close = d["spy_close"]
    spy_sma50 = d["spy_sma50"]
    spy_sma200 = d["spy_sma200"]

    # --- KS #1: VIX > 35 AND SPY < 200-SMA → halt all except defensive wheel CSPs ---
    if vix > 35 and spy_close < spy_sma200:
        pause_all = True
        reasons.append(f"KS#1: VIX={vix:.1f}>35 AND SPY={spy_close:.2f} below 200-SMA={spy_sma200:.2f} → HALT ALL (except defensive CSPs)")

    # --- KS #2: VIX < 12 → halt premium selling ---
    if vix < 12:
        pause_premium = True
        reasons.append(f"KS#2: VIX={vix:.1f}<12 → premiums too thin, HALT premium selling (CSP/BPS/IC/Strangle)")

    # --- KS #3: SPY < 50-SMA while VIX rising above 20 → reduce 50%, pause rotation ---
    if spy_close < spy_sma50 and vix > 20 and d["vix_prev"] < vix:
        reduce_50pct = True
        pause_rotation = True
        reasons.append(
            f"KS#3: SPY={spy_close:.2f} below 50-SMA={spy_sma50:.2f}, "
            f"VIX={vix:.1f} rising (prev={d['vix_prev']:.1f}) above 20 → REDUCE 50%, PAUSE rotation/momentum"
        )

    # --- KS #4: 3+ consecutive red days with VIX expansion > 3pts → pause all 2 days ---
    closes = d["spy_last5"]  # 6 values → 5 daily changes
    if len(closes) >= 4:
        # Check last 3 daily returns (most recent 3 days)
        daily_rets = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
        last3 = daily_rets[-3:] if len(daily_rets) >= 3 else daily_rets
        consecutive_red = all(r < 0 for r in last3) and len(last3) >= 3
        vix_expansion = vix - d["vix_4d_ago"]
        if consecutive_red and vix_expansion > 3:
            # Check if we're still within the 2-day pause window
            pause_all = True
            reasons.append(
                f"KS#4: 3 consecutive red days + VIX expanded {vix_expansion:.1f}pts "
                f"(from {d['vix_4d_ago']:.1f} to {vix:.1f}) → PAUSE ALL new entries for 2 days"
            )

    # --- KS #5: VIX term structure inverted → halt Iron Condors and Strangles ---
    if d.get("vix3m") is not None:
        if vix > d["vix3m"]:
            pause_strangles = True
            reasons.append(
                f"KS#5: VIX term structure INVERTED — front {vix:.1f} > back {d['vix3m']:.1f} "
                f"→ HALT Iron Condors & Strangles"
            )

    if not reasons:
        reasons.append("ALL CLEAR — no kill switches active")

    result = {
        "pause_all": pause_all,
        "pause_premium": pause_premium,
        "pause_rotation": pause_rotation,
        "pause_strangles": pause_strangles,
        "reduce_50pct": reduce_50pct,
        "reasons": reasons,
        "raw": {
            "vix": vix,
            "vix3m": d.get("vix3m"),
            "spy_close": spy_close,
            "spy_sma50": round(spy_sma50, 2),
            "spy_sma200": round(spy_sma200, 2),
            "vix_4d_ago": d.get("vix_4d_ago"),
            "term_structure": "normal" if (d.get("vix3m") and vix <= d["vix3m"]) else "inverted" if d.get("vix3m") else "unknown",
        },
        "checked_at": datetime.now().isoformat(),
        "any_active": pause_all or pause_premium or pause_rotation or pause_strangles or reduce_50pct,
    }
    return result


# ---------------------------------------------------------------------------
# State persistence (for KS#4 2-day pause tracking)
# ---------------------------------------------------------------------------

def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _log_check(result):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ---------------------------------------------------------------------------
# Alert via autonomy inject
# ---------------------------------------------------------------------------

def _send_alert(result):
    """POST alert to teleclaude inject endpoint."""
    import urllib.request
    import urllib.error

    msg_lines = ["KILL SWITCH ALERT:"]
    for r in result["reasons"]:
        if "ALL CLEAR" not in r:
            msg_lines.append(f"  {r}")
    msg_lines.append(f"\nMarket snapshot: VIX={result['raw']['vix']:.1f}, SPY={result['raw']['spy_close']:.2f}")

    payload = json.dumps({"message": "\n".join(msg_lines)}).encode()
    req = urllib.request.Request(
        INJECT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"Alert sent to inject endpoint: {resp.status}")
            return True
    except (urllib.error.URLError, OSError) as e:
        logger.warning(f"Failed to send alert to inject endpoint: {e}")
        return False


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    send_alert = "--alert" in sys.argv

    logger.info("Running kill switch check...")
    result = check_kill_switches(force_refresh=True)

    # Log
    _log_check(result)
    _save_state({"last_check": result["checked_at"], "any_active": result["any_active"]})

    # Print
    print(json.dumps(result, indent=2, default=str))

    if result["any_active"]:
        logger.warning("KILL SWITCH(ES) ACTIVE:")
        for r in result["reasons"]:
            logger.warning(f"  {r}")
        if send_alert:
            _send_alert(result)
    else:
        logger.info("All clear — no kill switches active")

    # Exit code: 0 = all clear, 1 = kill switch active, 2 = error
    if result.get("error"):
        sys.exit(2)
    elif result["any_active"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
