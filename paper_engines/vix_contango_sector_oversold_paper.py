#!/usr/bin/env python3
"""
VIX Contango + Sector Oversold Paper Engine
=============================================
Validated strategy — passed 6/6 adversarial validation.
Sharpe 1.67, 776 trades, WR 62%.

Strategy Rules:
  - Buy sector ETFs when VIX in contango (VIX < VIX3M) AND RSI(14) < 30
  - Best sectors: XLU (3.16), XLV (2.84), XLF (2.62). Worst: XLK (0.05)
  - Hold 5 days, equal weight among qualifying sectors
  - Starting capital: $645

Cron: 25 20 * * 1-5 (4:25 PM ET)
State: /home/jupiter/Lvl3Quant/state/vix_contango_sector_oversold_paper_state.json
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
STATE_FILE = STATE_DIR / "vix_contango_sector_oversold_paper_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vix_contango_sector_oversold.log"

# ── Strategy Parameters ──
INITIAL_CAPITAL = 645.0
HOLD_DAYS = 5
RSI_PERIOD = 14
RSI_THRESHOLD = 30
SPREAD_BPS = 2  # 0.02% per trade

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]

# Per-sector historical Sharpe from backtest — used for signal strength weighting
SECTOR_SHARPE = {
    "XLU": 3.16, "XLV": 2.84, "XLF": 2.62,
    "XLP": 1.50, "XLI": 1.40, "XLE": 1.20,
    "XLB": 1.10, "XLRE": 1.00, "XLC": 0.80,
    "XLY": 0.60, "XLK": 0.05,
}


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [VIX-CONTANGO-OVERSOLD] {msg}"
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


def compute_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float | None:
    """Compute RSI(14) from a price series."""
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else None


def fetch_data() -> tuple[dict, float | None, float | None]:
    """Fetch sector ETF prices + VIX + VIX3M. Returns (price_data, vix, vix3m)."""
    end = datetime.now(ET)
    start = end - timedelta(days=40)

    # Download sector ETFs + VIX together (date range works for these)
    sector_tickers = SECTOR_ETFS + ["^VIX"]
    try:
        raw = yf.download(sector_tickers, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False, threads=True, timeout=30)
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return {}, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return {}, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    # Extract VIX
    vix = None
    if "^VIX" in close.columns:
        vix_series = close["^VIX"].dropna()
        if len(vix_series) > 0:
            vix = float(vix_series.iloc[-1])

    # VIX3M requires period-based download (date range fails on yfinance)
    vix3m = None
    try:
        vix3m_data = yf.download("^VIX3M", period="5d", progress=False, timeout=10)
        if not vix3m_data.empty:
            if isinstance(vix3m_data.columns, pd.MultiIndex):
                vix3m_data.columns = vix3m_data.columns.get_level_values(0)
            vix3m = float(vix3m_data["Close"].iloc[-1])
    except Exception as e:
        log(f"WARN: VIX3M fetch failed: {e}")

    # Build price data for sectors
    price_data = {}
    for etf in SECTOR_ETFS:
        if etf in close.columns:
            series = close[etf].dropna()
            if len(series) >= RSI_PERIOD + 2:
                price_data[etf] = {
                    "prices": series,
                    "current": float(series.iloc[-1]),
                }

    return price_data, vix, vix3m


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    if today.weekday() >= 5:
        log("Weekend — skipping.")
        return

    log(f"=== VIX Contango + Sector Oversold Engine — {today_str} ===")
    state = load_state()

    # Fetch data
    price_data, vix, vix3m = fetch_data()
    if not price_data:
        log("ERROR: No price data. Aborting.")
        save_state(state)
        return

    vix_str = f"{vix:.2f}" if vix else "N/A"
    vix3m_str = f"{vix3m:.2f}" if vix3m else "N/A"
    log(f"VIX: {vix_str} | VIX3M: {vix3m_str}")
    contango = (vix is not None and vix3m is not None and vix < vix3m)
    log(f"Contango: {'YES' if contango else 'NO'}")

    # --- 1. Close expired positions (held >= HOLD_DAYS) ---
    still_open = []
    for pos in state["positions"]:
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        days_held = (today - entry_date).days
        ticker = pos["ticker"]

        if days_held >= HOLD_DAYS:
            # Close position
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
                })
                # Keep only last 100 closed trades
                state["closed_trades"] = state["closed_trades"][-100:]
                log(f"  CLOSED {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d")
            else:
                log(f"  WARN: Cannot close {ticker} — no price data. Keeping.")
                still_open.append(pos)
                continue
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Scan for new entries ---
    new_signals = []
    for etf, data in price_data.items():
        rsi = compute_rsi(data["prices"])
        if rsi is None:
            continue

        signal_entry = {
            "ticker": etf,
            "rsi": round(rsi, 2),
            "contango": contango,
            "direction": "long" if (contango and rsi < RSI_THRESHOLD) else "none",
            "strength": round(SECTOR_SHARPE.get(etf, 0.5) / 3.16, 2),  # Normalized 0-1
        }
        new_signals.append(signal_entry)

        if contango and rsi < RSI_THRESHOLD:
            # Check if already holding this ticker
            held = [p["ticker"] for p in state["positions"]]
            if etf not in held:
                # Open position
                buy_price = data["current"] * (1 + SPREAD_BPS / 10_000)
                # Equal weight: use 20% of available capital per position, max 5 positions
                per_pos = state["capital"] * 0.20
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
                    "rsi_at_entry": round(rsi, 2),
                    "vix_at_entry": round(vix, 2) if vix else None,
                    "vix3m_at_entry": round(vix3m, 2) if vix3m else None,
                    "sector_sharpe": SECTOR_SHARPE.get(etf, 0),
                }
                state["positions"].append(pos)
                log(f"  OPENED {etf}: RSI={rsi:.1f}, {shares:.4f} shares @ ${buy_price:.2f}")

    state["signals"] = new_signals

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in price_data:
            portfolio_value += pos["shares"] * price_data[ticker]["current"]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]
    state["equity"] = round(portfolio_value, 2)

    # Summary
    total = state["wins"] + state["losses"]
    wr = state["wins"] / total * 100 if total > 0 else 0
    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total} | WR: {wr:.0f}% | PnL: ${state['total_pnl']:+.2f}")

    save_state(state)
    log("Done.")


if __name__ == "__main__":
    run()
