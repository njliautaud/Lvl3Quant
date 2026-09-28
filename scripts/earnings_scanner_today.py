#!/usr/bin/env python3
"""
Earnings Scanner — Jul 29-30, 2026
Checks reporters, price action, volume, options affordability, and kill switch.
Saves results to state/earnings_watch.json
"""

import json
import os
import sys
from datetime import datetime, timedelta

import yfinance as yf
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

# Jul 30 reporters (tomorrow)
TOMORROW_REPORTERS = {
    "AAPL":  {"session": "PM", "est_eps": 1.89},
    "AMZN":  {"session": "PM", "est_eps": 1.82},
    "RBLX":  {"session": "PM", "est_eps": -0.34},
    "RIVN":  {"session": "PM", "est_eps": -0.79},
    "COIN":  {"session": "PM", "est_eps": 0.14},
    "RDDT":  {"session": "PM", "est_eps": 0.97},
    "MA":    {"session": "AM", "est_eps": 4.76},
}

# Jul 29 after-hours reporters (already reported)
YESTERDAY_AH = {
    "META":  {"session": "AH", "est_eps": None},
    "MSFT":  {"session": "AH", "est_eps": None},
    "QCOM":  {"session": "AH", "est_eps": None},
    "ARM":   {"session": "AH", "est_eps": None},
}

OPTIONS_AFFORDABLE_THRESHOLD = 100  # stock price < $100 => ~<$200/contract for ATM
UNUSUAL_VOLUME_RATIO = 1.5

def scan_ticker(ticker, meta):
    """Get price, returns, volume info for a single ticker."""
    try:
        t = yf.Ticker(ticker)

        # Get recent history (30 days for 20d return calc + volume avg)
        hist = t.history(period="1mo")
        if hist.empty or len(hist) < 2:
            return {"ticker": ticker, "error": "No data"}

        current_price = float(hist["Close"].iloc[-1])

        # 5d and 20d returns
        ret_5d = None
        ret_20d = None
        if len(hist) >= 6:
            ret_5d = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) * 100)
        if len(hist) >= 21:
            ret_20d = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) * 100)

        # Volume analysis
        today_vol = float(hist["Volume"].iloc[-1])
        avg_vol_20d = float(hist["Volume"].iloc[-21:].mean()) if len(hist) >= 21 else float(hist["Volume"].mean())
        vol_ratio = today_vol / avg_vol_20d if avg_vol_20d > 0 else 0
        unusual_volume = vol_ratio >= UNUSUAL_VOLUME_RATIO

        # Options affordability
        affordable = current_price < OPTIONS_AFFORDABLE_THRESHOLD
        approx_atm_cost = round(current_price * 0.02 * 100, 0)  # rough ~2% ATM premium estimate

        result = {
            "ticker": ticker,
            "session": meta.get("session"),
            "est_eps": meta.get("est_eps"),
            "current_price": round(current_price, 2),
            "return_5d_pct": round(ret_5d, 2) if ret_5d else None,
            "return_20d_pct": round(ret_20d, 2) if ret_20d else None,
            "volume_today": int(today_vol),
            "avg_volume_20d": int(avg_vol_20d),
            "volume_ratio": round(vol_ratio, 2),
            "unusual_volume": unusual_volume,
            "options_affordable": affordable,
            "approx_atm_contract_cost": approx_atm_cost,
        }
        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def check_kill_switch():
    """Check VIX > 20 AND SPY < 50-SMA => kill switch ON."""
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        vix_level = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        if spy_hist.empty or len(spy_hist) < 50:
            return {"vix": vix_level, "spy_price": None, "spy_50sma": None, "kill_switch": "UNKNOWN"}

        spy_price = float(spy_hist["Close"].iloc[-1])
        spy_50sma = float(spy_hist["Close"].iloc[-50:].mean())

        kill_switch = (vix_level is not None and vix_level > 20) and (spy_price < spy_50sma)

        return {
            "vix": round(vix_level, 2) if vix_level else None,
            "spy_price": round(spy_price, 2),
            "spy_50sma": round(spy_50sma, 2),
            "kill_switch": "ON — PAUSED" if kill_switch else "OFF — CLEAR",
        }
    except Exception as e:
        return {"error": str(e), "kill_switch": "UNKNOWN"}


def main():
    print("=" * 70)
    print("EARNINGS SCANNER — Jul 29-30, 2026")
    print("=" * 70)

    # ── Kill Switch ─────────────────────────────────────────────────────
    print("\n--- KILL SWITCH STATUS ---")
    ks = check_kill_switch()
    print(f"  VIX: {ks.get('vix')}  |  SPY: {ks.get('spy_price')}  |  50-SMA: {ks.get('spy_50sma')}")
    print(f"  Kill Switch: {ks.get('kill_switch')}")

    # ── Yesterday's AH reporters (already reported) ─────────────────────
    print("\n--- YESTERDAY AH (Jul 29 — already reported) ---")
    yesterday_results = []
    for ticker, meta in YESTERDAY_AH.items():
        r = scan_ticker(ticker, meta)
        yesterday_results.append(r)
        if "error" not in r:
            vol_flag = " ** UNUSUAL VOL **" if r["unusual_volume"] else ""
            print(f"  {ticker:6s}  ${r['current_price']:>8.2f}  5d: {r['return_5d_pct']:>+6.2f}%  20d: {r['return_20d_pct']:>+6.2f}%  VolRatio: {r['volume_ratio']:.1f}x{vol_flag}")
        else:
            print(f"  {ticker:6s}  ERROR: {r['error']}")

    # ── Tomorrow's reporters ────────────────────────────────────────────
    print("\n--- TOMORROW (Jul 30) REPORTERS ---")
    tomorrow_results = []
    affordable_tickers = []

    for ticker, meta in TOMORROW_REPORTERS.items():
        r = scan_ticker(ticker, meta)
        tomorrow_results.append(r)
        if "error" not in r:
            vol_flag = " ** UNUSUAL VOL **" if r["unusual_volume"] else ""
            afford = "AFFORDABLE" if r["options_affordable"] else f"EXPENSIVE (~${r['approx_atm_contract_cost']:.0f}/contract)"
            eps_str = f"Est EPS: {meta['est_eps']}" if meta['est_eps'] else ""
            print(f"  {ticker:6s}  ${r['current_price']:>8.2f}  {meta['session']}  {eps_str}")
            print(f"         5d: {r['return_5d_pct']:>+6.2f}%  20d: {r['return_20d_pct']:>+6.2f}%  VolRatio: {r['volume_ratio']:.1f}x{vol_flag}")
            print(f"         Options: {afford}")
            if r["options_affordable"]:
                affordable_tickers.append(ticker)
        else:
            print(f"  {ticker:6s}  ERROR: {r['error']}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n--- PEAD PLAY SUMMARY ---")
    print(f"  Affordable for options (<${OPTIONS_AFFORDABLE_THRESHOLD}): {', '.join(affordable_tickers) if affordable_tickers else 'NONE'}")
    unusual = [r["ticker"] for r in tomorrow_results if r.get("unusual_volume")]
    print(f"  Unusual volume (>{UNUSUAL_VOLUME_RATIO}x avg): {', '.join(unusual) if unusual else 'NONE'}")
    print(f"  Kill switch: {ks.get('kill_switch')}")

    # ── Save to JSON ────────────────────────────────────────────────────
    output = {
        "scan_time": datetime.now().isoformat(),
        "kill_switch": ks,
        "tomorrow_jul30": tomorrow_results,
        "yesterday_ah_jul29": yesterday_results,
        "affordable_for_pead": affordable_tickers,
        "unusual_volume_flags": unusual,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state", "earnings_watch.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
