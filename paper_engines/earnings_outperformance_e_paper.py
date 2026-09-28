#!/usr/bin/env python3
"""
Earnings Outperformance E Paper Engine
========================================
Validated strategy — passed 6/6 adversarial validation.
Sharpe 2.91, 674 trades, WR 62%.

Strategy Rules:
  - Long sector ETFs that outperformed SPY by >1% over a 10-day earnings window
  - Hold 5 days after identifying the outperformance
  - Equal weight among qualifying sectors
  - Starting capital: $645

Cron: 25 20 * * 1-5 (4:25 PM ET)
State: /home/jupiter/Lvl3Quant/state/earnings_outperformance_e_paper_state.json
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "earnings_outperformance_e_paper_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "earnings_outperformance_e.log"

# ── Strategy Parameters ──
INITIAL_CAPITAL = 645.0
HOLD_DAYS = 5
EARNINGS_WINDOW = 10     # 10-day window to measure outperformance
OUTPERF_THRESHOLD = 0.01  # >1% outperformance vs SPY
SPREAD_BPS = 2

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [EARNINGS-OUTPERF-E] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "positions": [],
        "closed_trades": [],
        "signals": [],
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "total_pnl": 0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "created": datetime.now(ET).isoformat(),
        "last_updated": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S.%f")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def fetch_data() -> dict:
    """Fetch sector ETF + SPY prices. Returns {ticker: {prices: Series, current: float}}."""
    tickers = SECTOR_ETFS + ["SPY"]
    end = datetime.now(ET)
    start = end - timedelta(days=30)  # Need ~15 trading days

    try:
        raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True, timeout=30)
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return {}

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return {}

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    price_data = {}
    for t in tickers:
        if t in close.columns:
            series = close[t].dropna()
            if len(series) >= EARNINGS_WINDOW + 2:
                price_data[t] = {
                    "prices": series,
                    "current": float(series.iloc[-1]),
                }

    return price_data


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend — skipping.")
        return

    log(f"=== Earnings Outperformance E Engine — {today_str} ===")
    state = load_state()

    # Fetch data
    price_data = fetch_data()
    if "SPY" not in price_data:
        log("ERROR: No SPY data available. Aborting.")
        save_state(state)
        return

    if len(price_data) < 6:
        log(f"ERROR: Only {len(price_data)} tickers with data. Need more. Aborting.")
        save_state(state)
        return

    spy_prices = price_data["SPY"]["prices"]

    # --- 1. Compute 10-day relative performance vs SPY ---
    spy_ret_10d = float(spy_prices.iloc[-1] / spy_prices.iloc[-(EARNINGS_WINDOW + 1)] - 1)
    log(f"SPY 10d return: {spy_ret_10d*100:+.2f}%")

    outperformers = {}
    new_signals = []

    for etf in SECTOR_ETFS:
        if etf not in price_data:
            continue

        prices = price_data[etf]["prices"]
        if len(prices) < EARNINGS_WINDOW + 1:
            continue

        etf_ret_10d = float(prices.iloc[-1] / prices.iloc[-(EARNINGS_WINDOW + 1)] - 1)
        excess_return = etf_ret_10d - spy_ret_10d
        qualifies = excess_return > OUTPERF_THRESHOLD

        signal = {
            "ticker": etf,
            "direction": "long" if qualifies else "none",
            "strength": round(min(1.0, excess_return / 0.05), 2) if qualifies else 0,
            "ret_10d": round(etf_ret_10d * 100, 2),
            "spy_ret_10d": round(spy_ret_10d * 100, 2),
            "excess_return": round(excess_return * 100, 2),
        }
        new_signals.append(signal)

        if qualifies:
            outperformers[etf] = excess_return
            log(f"  {etf}: 10d={etf_ret_10d*100:+.2f}%, excess={excess_return*100:+.2f}% -- QUALIFIES")
        else:
            log(f"  {etf}: 10d={etf_ret_10d*100:+.2f}%, excess={excess_return*100:+.2f}%")

    state["signals"] = new_signals

    # --- 2. Close expired positions ---
    still_open = []
    for pos in state["positions"]:
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        days_held = (today - entry_date).days
        ticker = pos["ticker"]

        if days_held >= HOLD_DAYS:
            if ticker in price_data:
                exit_price = price_data[ticker]["current"] * (1 - SPREAD_BPS / 10_000)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100

                state["capital"] += pos["shares"] * exit_price
                state["total_pnl"] += pnl
                state["n_trades"] += 1
                if pnl >= 0:
                    state["wins"] += 1
                else:
                    state["losses"] += 1

                state["closed_trades"].append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": today_str,
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_price, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "days_held": days_held,
                    "excess_return_at_entry": pos.get("excess_return_at_entry"),
                })
                state["closed_trades"] = state["closed_trades"][-100:]
                log(f"  CLOSED {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d")
            else:
                log(f"  WARN: Cannot close {ticker} — no price. Keeping.")
                still_open.append(pos)
                continue
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 3. Open new positions for outperformers ---
    held_tickers = [p["ticker"] for p in state["positions"]]
    # Sort outperformers by excess return (strongest first)
    sorted_outperf = sorted(outperformers.items(), key=lambda x: x[1], reverse=True)

    for etf, excess_ret in sorted_outperf:
        if etf in held_tickers:
            log(f"  SKIP {etf}: already holding")
            continue

        if etf not in price_data:
            continue

        buy_price = price_data[etf]["current"] * (1 + SPREAD_BPS / 10_000)
        # Allocate ~25% per position (max 4 concurrent)
        n_open = len(state["positions"])
        if n_open >= 4:
            log(f"  SKIP {etf}: max 4 positions already open")
            break

        per_pos = state["capital"] * 0.25
        if per_pos < 20:
            log(f"  SKIP {etf}: insufficient capital (${state['capital']:.0f})")
            continue

        shares = per_pos / buy_price
        cost = shares * buy_price
        state["capital"] -= cost

        pos = {
            "ticker": etf,
            "shares": round(shares, 6),
            "entry_price": round(buy_price, 4),
            "entry_date": now.isoformat(),
            "excess_return_at_entry": round(excess_ret * 100, 2),
        }
        state["positions"].append(pos)
        log(f"  OPENED {etf}: excess={excess_ret*100:+.2f}%, "
            f"{shares:.4f} shares @ ${buy_price:.2f}")

    # --- 4. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in price_data:
            portfolio_value += pos["shares"] * price_data[ticker]["current"]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]
    state["equity"] = round(portfolio_value, 2)

    total = state["wins"] + state["losses"]
    wr = state["wins"] / total * 100 if total > 0 else 0
    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total} | WR: {wr:.0f}% | PnL: ${state['total_pnl']:+.2f}")

    save_state(state)
    log("Done.")


if __name__ == "__main__":
    run()
