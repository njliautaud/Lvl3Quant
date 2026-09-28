#!/usr/bin/env python3
"""
RSI Divergence C — Paper Trading Engine
=========================================

Strategy #8 (6/6 adversarial gates passed)

BUY quality mega-caps when:
  - Price makes new 20-day low (or lower low vs 20 days ago)
  - RSI makes HIGHER low (bullish divergence)
  - Volume declining during the dip (20d avg vol < 40d avg vol)
  - Stock is >5% below 52-week high
Hold 10 days fixed.

Capital: $10,000 | Max $500/position | Max 3 concurrent

Usage:
  python3 paper_engines/rsi_divergence_paper.py
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

STATE_FILE = STATE_DIR / "rsi_divergence_paper_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "rsi_divergence_paper.log"),
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


def check_signals(raw_df):
    """Check for RSI bullish divergence signals."""
    signals = []
    for ticker in QUALITY_STOCKS:
        try:
            if len(QUALITY_STOCKS) > 1 and isinstance(raw_df.columns, pd.MultiIndex):
                close = raw_df[(ticker, "Close")].dropna()
                volume = raw_df[(ticker, "Volume")].dropna()
                high = raw_df[(ticker, "High")].dropna()
            else:
                close = raw_df["Close"].dropna()
                volume = raw_df["Volume"].dropna()
                high = raw_df["High"].dropna()

            if len(close) < 60:
                continue

            rsi = compute_rsi(close)
            price_now = close.iloc[-1]
            rsi_now = rsi.iloc[-1]

            # Price makes new 20-day low or lower low vs 20 days ago
            price_20d_ago = close.iloc[-20] if len(close) >= 20 else close.iloc[0]
            price_20d_low = close.iloc[-20:].min()
            price_at_low = price_now <= price_20d_low * 1.005  # within 0.5% of 20d low

            if not price_at_low:
                continue

            # RSI makes HIGHER low (bullish divergence)
            rsi_20d_ago = rsi.iloc[-20] if len(rsi) >= 20 else rsi.iloc[0]
            rsi_20d_min = rsi.iloc[-20:].min()
            # Find RSI at the price low point
            price_low_idx = close.iloc[-20:].idxmin()
            rsi_at_price_low = rsi.loc[price_low_idx] if price_low_idx in rsi.index else rsi_20d_min

            # Bullish divergence: price low but RSI higher than previous trough
            if rsi_now <= rsi_at_price_low:
                continue  # No divergence

            # Volume declining during dip
            vol_20d = volume.iloc[-20:].mean()
            vol_40d = volume.iloc[-40:].mean() if len(volume) >= 40 else volume.mean()
            if vol_20d >= vol_40d:
                continue

            # Stock >5% below 52-week high
            high_52w = high.iloc[-252:].max() if len(high) >= 252 else high.max()
            pct_below_high = 1 - (price_now / high_52w)
            if pct_below_high < 0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": round(float(price_now), 2),
                "rsi": round(float(rsi_now), 1),
                "pct_below_52w_high": round(float(pct_below_high * 100), 1),
                "reason": f"RSI div: price near 20d low, RSI higher ({rsi_now:.0f} vs {rsi_at_price_low:.0f}), vol declining",
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
    log.info("RSI Divergence C — Paper Engine")
    log.info("=" * 60)

    state = load_state()
    today_str = date.today().isoformat()

    # Idempotency: skip if already run today
    if state.get("last_run_date") == today_str:
        log.info(f"Already ran today ({today_str}). Skipping.")
        return

    # Download data
    raw_df = _yf_download(QUALITY_STOCKS, period="120d")
    if raw_df is None:
        log.error("Failed to download data. Aborting.")
        return

    # Process exits first
    process_exits(state, raw_df)

    # Check signals
    signals = check_signals(raw_df)
    log.info(f"  Signals found: {len(signals)}")
    for s in signals:
        log.info(f"    {s['ticker']}: RSI={s['rsi']}, {s['pct_below_52w_high']}% below 52w high")

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
