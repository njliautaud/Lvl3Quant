#!/usr/bin/env python3
"""
Bond Yield Signal B — Paper Trading Engine
=============================================

Strategy #9 (6/6 adversarial gates passed)

BUY quality stocks when:
  - 10Y yield (^TNX) dropped >10 bps over last 5 trading days (flight-to-safety ending)
  - Stock is >5% below its 20-day SMA
Hold 10 days fixed.

Capital: $10,000 | Max $500/position | Max 3 concurrent

Usage:
  python3 paper_engines/bond_yield_paper.py
"""

import json
import logging
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "bond_yield_paper_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bond_yield_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CAPITAL_INITIAL = 10000
MAX_PER_POSITION = 500
MAX_CONCURRENT = 3
HOLD_DAYS = 10

QUALITY_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM",
    "JNJ", "UNH", "PG", "HD", "MA", "ABBV", "KO", "PEP", "COST", "LIN",
    "CRM", "AVGO", "TMO", "MRK", "ACN",
]


def _yf_download(tickers, period="120d", retries=3):
    """Download with retry logic."""
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(tickers, period=period, progress=False, group_by="ticker")
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"yfinance attempt {attempt+1} failed: {e}")
        if attempt < retries - 1:
            time.sleep(3)
    log.error("yfinance download failed after retries")
    return None


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": CAPITAL_INITIAL,
        "positions": [],
        "closed_trades": [],
        "last_run_date": None,
        "created": datetime.now().isoformat(),
    }


def save_state(state):
    state["updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _flatten_df(df):
    """Flatten MultiIndex columns from yfinance single-ticker download."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    return df


def check_yield_drop(tnx_df):
    """Check if 10Y yield dropped >10 bps over last 5 trading days."""
    try:
        tnx_df = _flatten_df(tnx_df)
        close = tnx_df["Close"].dropna()
        if len(close) < 6:
            return False, 0.0
        yield_now = float(close.iloc[-1])
        yield_5d_ago = float(close.iloc[-6])  # 5 trading days back
        change = yield_now - yield_5d_ago
        return change < -0.10, round(change, 3)
    except Exception as e:
        log.warning(f"Yield check failed: {e}")
        return False, 0.0


def check_signals(raw_df, yield_dropped):
    """Check for buy signals: stock >5% below 20d SMA when yields dropping."""
    signals = []
    if not yield_dropped:
        return signals

    for ticker in QUALITY_STOCKS:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                close = raw_df[(ticker, "Close")].dropna()
            else:
                close = raw_df["Close"].dropna()

            if len(close) < 25:
                continue

            price_now = float(close.iloc[-1])
            sma_20 = float(close.iloc[-20:].mean())

            pct_below_sma = 1 - (price_now / sma_20)
            if pct_below_sma < 0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": round(price_now, 2),
                "pct_below_sma20": round(pct_below_sma * 100, 1),
                "reason": f"Bond yield drop + {pct_below_sma*100:.1f}% below 20d SMA",
            })
        except Exception as e:
            log.debug(f"Signal check failed for {ticker}: {e}")
    return signals


def process_exits(state, raw_df):
    """Exit positions held >= HOLD_DAYS trading days."""
    today = date.today()
    remaining = []
    for pos in state["positions"]:
        entry_date = datetime.fromisoformat(pos["entry_date"]).date()
        days_held = np.busday_count(entry_date, today)
        if days_held >= HOLD_DAYS:
            ticker = pos["ticker"]
            try:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    exit_price = float(raw_df[(ticker, "Close")].dropna().iloc[-1])
                else:
                    exit_price = float(raw_df["Close"].dropna().iloc[-1])
            except Exception:
                exit_price = pos["entry_price"]

            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            state["capital"] += pos["shares"] * exit_price
            trade = {
                **pos,
                "exit_date": today.isoformat(),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "days_held": int(days_held),
            }
            state["closed_trades"].append(trade)
            log.info(f"  EXIT {ticker}: ${pnl:+.2f} ({days_held}d hold)")
        else:
            remaining.append(pos)
    state["positions"] = remaining


def process_entries(state, signals, raw_df):
    """Open new positions from signals."""
    open_count = len(state["positions"])
    open_tickers = {p["ticker"] for p in state["positions"]}
    today = date.today()

    for sig in signals:
        if open_count >= MAX_CONCURRENT:
            break
        if sig["ticker"] in open_tickers:
            continue

        price = sig["price"]
        if price <= 0:
            continue
        shares = int(MAX_PER_POSITION / price)
        if shares < 1:
            continue
        cost = shares * price
        if cost > state["capital"]:
            continue

        state["capital"] -= cost
        pos = {
            "ticker": sig["ticker"],
            "entry_date": today.isoformat(),
            "entry_price": price,
            "shares": shares,
            "reason": sig["reason"],
        }
        state["positions"].append(pos)
        open_tickers.add(sig["ticker"])
        open_count += 1
        log.info(f"  ENTRY {sig['ticker']}: {shares} shares @ ${price:.2f} — {sig['reason']}")


def compute_metrics(state):
    """Compute portfolio metrics."""
    trades = state["closed_trades"]
    if not trades:
        return {"total_trades": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "sharpe": round(np.mean(pnls) / np.std(pnls), 2) if len(pnls) > 1 and np.std(pnls) > 0 else 0,
    }


def main():
    log.info("=" * 60)
    log.info("Bond Yield Signal B — Paper Engine")
    log.info("=" * 60)

    state = load_state()
    today_str = date.today().isoformat()

    if state.get("last_run_date") == today_str:
        log.info(f"Already ran today ({today_str}). Skipping.")
        return

    # Download 10Y yield
    import yfinance as yf
    tnx_df = _yf_download("^TNX", period="30d")
    if tnx_df is None:
        log.error("Failed to download ^TNX. Aborting.")
        return

    yield_dropped, yield_change = check_yield_drop(tnx_df)
    log.info(f"  10Y yield 5d change: {yield_change:+.3f}% | Signal: {'YES' if yield_dropped else 'NO'}")

    # Download stock data
    raw_df = _yf_download(QUALITY_STOCKS, period="60d")
    if raw_df is None:
        log.error("Failed to download stock data. Aborting.")
        return

    # Process exits first
    process_exits(state, raw_df)

    # Check signals
    signals = check_signals(raw_df, yield_dropped)
    log.info(f"  Signals found: {len(signals)}")
    for s in signals:
        log.info(f"    {s['ticker']}: {s['pct_below_sma20']}% below 20d SMA")

    # Process entries
    process_entries(state, signals, raw_df)

    # Compute equity
    equity = state["capital"]
    for pos in state["positions"]:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                cur = float(raw_df[(pos["ticker"], "Close")].dropna().iloc[-1])
            else:
                cur = float(raw_df["Close"].dropna().iloc[-1])
        except Exception:
            cur = pos["entry_price"]
        equity += pos["shares"] * cur

    metrics = compute_metrics(state)
    state["last_run_date"] = today_str
    state["last_equity"] = round(equity, 2)
    save_state(state)

    log.info(f"  SUMMARY | Date: {today_str} | Equity: ${equity:,.2f} | "
             f"Positions: {len(state['positions'])} | Closed: {metrics['total_trades']} | "
             f"WR: {metrics.get('win_rate', 0)}% | Sharpe: {metrics.get('sharpe', 0)}")


if __name__ == "__main__":
    main()
