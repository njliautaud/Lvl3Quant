#!/usr/bin/env python3
"""
IV-RV Gap Entry — Paper Trading Engine
=========================================

Strategy #10 (6/6 adversarial gates passed)

BUY quality stocks when:
  - VIX > SPY 20-day realized vol (annualized) by 5+ points
    (implied fear exceeds actual movement — market overpricing risk)
  - Stock is >5% below 52-week high
  - RSI < 40
Hold 10 days fixed.

Capital: $10,000 | Max $500/position | Max 3 concurrent

Usage:
  python3 paper_engines/iv_rv_gap_paper.py
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

STATE_FILE = STATE_DIR / "iv_rv_gap_paper_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "iv_rv_gap_paper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CAPITAL_INITIAL = 10000
MAX_PER_POSITION = 500
MAX_CONCURRENT = 3
HOLD_DAYS = 10
IV_RV_GAP_MIN = 5.0  # VIX must exceed realized vol by 5+ points

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


def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def check_iv_rv_gap(vix_df, spy_df):
    """Check if VIX > SPY 20d realized vol (annualized) by 5+ points."""
    try:
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df = vix_df.droplevel("Ticker", axis=1)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df = spy_df.droplevel("Ticker", axis=1)
        vix_close = vix_df["Close"].dropna()
        spy_close = spy_df["Close"].dropna()

        if len(vix_close) < 2 or len(spy_close) < 21:
            return False, 0.0, 0.0, 0.0

        vix_now = float(vix_close.iloc[-1])

        # SPY 20-day realized vol, annualized
        spy_returns = spy_close.pct_change().dropna()
        rv_20d = float(spy_returns.iloc[-20:].std() * np.sqrt(252) * 100)

        gap = vix_now - rv_20d
        return gap >= IV_RV_GAP_MIN, round(gap, 2), round(vix_now, 2), round(rv_20d, 2)
    except Exception as e:
        log.warning(f"IV-RV gap check failed: {e}")
        return False, 0.0, 0.0, 0.0


def check_signals(raw_df, gap_active):
    """Check for buy signals: stock >5% below 52w high, RSI < 40, when IV-RV gap active."""
    signals = []
    if not gap_active:
        return signals

    for ticker in QUALITY_STOCKS:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                close = raw_df[(ticker, "Close")].dropna()
                high = raw_df[(ticker, "High")].dropna()
            else:
                close = raw_df["Close"].dropna()
                high = raw_df["High"].dropna()

            if len(close) < 30:
                continue

            price_now = float(close.iloc[-1])
            rsi = compute_rsi(close)
            rsi_now = float(rsi.iloc[-1])

            if rsi_now >= 40:
                continue

            # >5% below 52-week high
            high_52w = float(high.max())  # Use all available data up to 252d
            pct_below = 1 - (price_now / high_52w)
            if pct_below < 0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": round(price_now, 2),
                "rsi": round(rsi_now, 1),
                "pct_below_52w": round(pct_below * 100, 1),
                "reason": f"IV-RV gap active, RSI={rsi_now:.0f}, {pct_below*100:.1f}% below 52w high",
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
    log.info("IV-RV Gap Entry — Paper Engine")
    log.info("=" * 60)

    state = load_state()
    today_str = date.today().isoformat()

    if state.get("last_run_date") == today_str:
        log.info(f"Already ran today ({today_str}). Skipping.")
        return

    # Download VIX and SPY
    vix_df = _yf_download("^VIX", period="60d")
    spy_df = _yf_download("SPY", period="60d")
    if vix_df is None or spy_df is None:
        log.error("Failed to download VIX/SPY. Aborting.")
        return

    gap_active, gap, vix_val, rv_val = check_iv_rv_gap(vix_df, spy_df)
    log.info(f"  VIX: {vix_val} | SPY 20d RV: {rv_val} | Gap: {gap:+.1f} pts | Signal: {'YES' if gap_active else 'NO'}")

    # Download stock data
    raw_df = _yf_download(QUALITY_STOCKS, period="120d")
    if raw_df is None:
        log.error("Failed to download stock data. Aborting.")
        return

    # Process exits first
    process_exits(state, raw_df)

    # Check signals
    signals = check_signals(raw_df, gap_active)
    log.info(f"  Signals found: {len(signals)}")
    for s in signals:
        log.info(f"    {s['ticker']}: RSI={s['rsi']}, {s['pct_below_52w']}% below 52w high")

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
