#!/usr/bin/env python3
"""
Options Flow Collector
Captures options chain snapshots (volume, OI, IV, put/call ratios) for a
quality universe and appends to a historical CSV.

Run twice daily via cron: 9:45 AM ET and 3:45 PM ET
  45 9,15 * * 1-5 /usr/bin/python3 /home/jupiter/Lvl3Quant/strategies/options_flow_collector.py

Output: /home/jupiter/Lvl3Quant/data/options_flow_history.csv
"""

import os
import sys
import csv
import math
import traceback
from datetime import datetime, timezone

import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO",
    # Financials / health / staples
    "JPM", "UNH", "LLY", "V", "MA", "ABBV", "COST", "HD",
    "PG", "JNJ", "MRK", "PEP", "KO", "WMT",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP",
    "XLB", "XLRE", "XLU",
    # Benchmark
    "SPY",
]

OUTPUT_CSV = "/home/jupiter/Lvl3Quant/data/options_flow_history.csv"

COLUMNS = [
    "date",
    "timestamp",
    "ticker",
    "price",
    "daily_pct_change",
    "call_volume",
    "put_volume",
    "volume_pc_ratio",
    "call_oi",
    "put_oi",
    "oi_pc_ratio",
    "atm_iv",
    "hv_20d",
    "iv_hv_spread",
    "relative_volume",
]

HV_WINDOW = 20
AVG_VOL_WINDOW = 30  # days for relative-volume baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    if b is None or b == 0:
        return None
    return round(a / b, 4)


def _annualized_hv(closes, window=HV_WINDOW):
    """20-day realized (historical) volatility, annualized."""
    if closes is None or len(closes) < window + 1:
        return None
    log_rets = np.diff(np.log(closes[-(window + 1) :]))
    return round(float(np.std(log_rets) * math.sqrt(252)), 4)


def _nearest_atm_iv(chain_df, spot):
    """Pick the strike closest to spot and return its IV."""
    if chain_df is None or chain_df.empty or spot is None:
        return None
    try:
        chain_df = chain_df.dropna(subset=["impliedVolatility"])
        if chain_df.empty:
            return None
        idx = (chain_df["strike"] - spot).abs().idxmin()
        return round(float(chain_df.loc[idx, "impliedVolatility"]), 4)
    except Exception:
        return None


def collect_one(ticker_str: str) -> dict | None:
    """Collect options flow data for a single ticker. Returns dict or None."""
    try:
        tk = yf.Ticker(ticker_str)

        # --- Price / change ---
        hist = tk.history(period="2mo", auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return None

        price = round(float(hist["Close"].iloc[-1]), 2)
        prev_close = float(hist["Close"].iloc[-2])
        daily_pct = round((price - prev_close) / prev_close * 100, 2)

        # --- Historical volatility ---
        closes = hist["Close"].values
        hv_20 = _annualized_hv(closes)

        # --- Historical average total options volume (for relative volume) ---
        # We'll compute this from the chain snapshot below; approximate with
        # today's chain volumes vs a scalar (yfinance doesn't give historical
        # options volume time-series, so relative_volume is today's total
        # options volume vs the 30-day average *equity* volume as a rough proxy).
        equity_vol_series = hist["Volume"].values
        avg_equity_vol = (
            float(np.mean(equity_vol_series[-AVG_VOL_WINDOW:]))
            if len(equity_vol_series) >= AVG_VOL_WINDOW
            else float(np.mean(equity_vol_series))
        )
        today_equity_vol = float(equity_vol_series[-1]) if len(equity_vol_series) > 0 else None

        # --- Options chain (nearest expiry) ---
        expiries = tk.options
        if not expiries or len(expiries) == 0:
            # No options (some ETFs). Return equity-only row.
            return {
                "ticker": ticker_str,
                "price": price,
                "daily_pct_change": daily_pct,
                "call_volume": None,
                "put_volume": None,
                "volume_pc_ratio": None,
                "call_oi": None,
                "put_oi": None,
                "oi_pc_ratio": None,
                "atm_iv": None,
                "hv_20d": hv_20,
                "iv_hv_spread": None,
                "relative_volume": _safe_div(today_equity_vol, avg_equity_vol),
            }

        nearest_expiry = expiries[0]
        chain = tk.option_chain(nearest_expiry)
        calls = chain.calls
        puts = chain.puts

        call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
        put_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
        call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

        # ATM IV: average of nearest-ATM call IV and put IV
        call_atm_iv = _nearest_atm_iv(calls, price)
        put_atm_iv = _nearest_atm_iv(puts, price)
        if call_atm_iv is not None and put_atm_iv is not None:
            atm_iv = round((call_atm_iv + put_atm_iv) / 2, 4)
        else:
            atm_iv = call_atm_iv or put_atm_iv

        iv_hv_spread = (
            round(atm_iv - hv_20, 4) if atm_iv is not None and hv_20 is not None else None
        )

        # Relative volume: today's equity volume vs 30-day average
        rel_vol = _safe_div(today_equity_vol, avg_equity_vol)

        return {
            "ticker": ticker_str,
            "price": price,
            "daily_pct_change": daily_pct,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "volume_pc_ratio": _safe_div(put_vol, call_vol),
            "call_oi": call_oi,
            "put_oi": put_oi,
            "oi_pc_ratio": _safe_div(put_oi, call_oi),
            "atm_iv": atm_iv,
            "hv_20d": hv_20,
            "iv_hv_spread": iv_hv_spread,
            "relative_volume": rel_vol,
        }

    except Exception as e:
        print(f"  ERROR {ticker_str}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"Options Flow Collector  |  {ts_str}")
    print(f"Universe: {len(UNIVERSE)} tickers")
    print("-" * 60)

    # Check if CSV exists; write header if not
    write_header = not os.path.exists(OUTPUT_CSV)

    rows = []
    ok = 0
    fail = 0

    for sym in UNIVERSE:
        print(f"  {sym} ... ", end="", flush=True)
        row = collect_one(sym)
        if row is None:
            print("SKIP (no data)")
            fail += 1
            continue
        row["date"] = date_str
        row["timestamp"] = ts_str
        rows.append(row)
        pc = row.get("volume_pc_ratio")
        iv = row.get("atm_iv")
        pc_str = f"P/C={pc:.2f}" if pc is not None else "P/C=n/a"
        iv_str = f"IV={iv:.1%}" if iv is not None else "IV=n/a"
        print(f"${row['price']:>8.2f}  {row['daily_pct_change']:>+6.2f}%  {pc_str}  {iv_str}")
        ok += 1

    # Append to CSV
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("-" * 60)
    print(f"Done: {ok} collected, {fail} skipped. Appended to {OUTPUT_CSV}")
    existing_lines = 0
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            existing_lines = sum(1 for _ in f) - 1  # minus header
    print(f"Total rows in history file: {existing_lines}")


if __name__ == "__main__":
    main()
