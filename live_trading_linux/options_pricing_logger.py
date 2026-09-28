"""
Options Pricing Logger (HC #702)
================================
Logs every options pricing event from paper engines so we can later
compare BS estimates vs real Alpaca market quotes.

Writes to: data/options_pricing_audit/pricing_log.csv

Columns:
  timestamp, engine, ticker, spot, strike, expiry, option_type,
  dte_days, iv_used, bs_price, alpaca_mid, alpaca_bid, alpaca_ask,
  alpaca_iv, alpaca_delta, source, action (entry/exit/mtm)

Usage:
    from live_trading_linux.options_pricing_logger import log_option_price
    log_option_price(
        engine="wheel_ic", ticker="AAPL", spot=195.0, strike=190.0,
        expiry="2026-07-18", option_type="put", iv_used=0.25,
        bs_price=1.23, alpaca_mid=1.45, alpaca_bid=1.30, alpaca_ask=1.60,
        alpaca_iv=0.28, alpaca_delta=-0.25, source="alpaca", action="entry"
    )
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "data" / "options_pricing_audit"
LOG_FILE = LOG_DIR / "pricing_log.csv"
_lock = threading.Lock()

COLUMNS = [
    "timestamp", "engine", "ticker", "spot", "strike", "expiry",
    "option_type", "dte_days", "iv_used", "bs_price",
    "alpaca_mid", "alpaca_bid", "alpaca_ask", "alpaca_iv", "alpaca_delta",
    "source", "action"
]


def _ensure_file():
    """Create log dir and CSV header if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)


def log_option_price(
    engine: str,
    ticker: str,
    spot: float,
    strike: float,
    expiry: str,
    option_type: str,
    iv_used: float = 0.0,
    bs_price: float = 0.0,
    alpaca_mid: float = 0.0,
    alpaca_bid: float = 0.0,
    alpaca_ask: float = 0.0,
    alpaca_iv: float = 0.0,
    alpaca_delta: float = 0.0,
    source: str = "bs_only",
    action: str = "entry",
):
    """Log a single options pricing observation."""
    try:
        from datetime import date as dt_date
        # Calculate DTE
        if isinstance(expiry, str):
            exp_date = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        elif isinstance(expiry, dt_date):
            exp_date = expiry
        else:
            exp_date = datetime.today().date()
        dte = (exp_date - datetime.today().date()).days

        row = [
            datetime.utcnow().isoformat(timespec='seconds'),
            engine,
            ticker,
            f"{spot:.2f}",
            f"{strike:.2f}",
            str(expiry)[:10],
            option_type,
            dte,
            f"{iv_used:.4f}",
            f"{bs_price:.4f}",
            f"{alpaca_mid:.4f}",
            f"{alpaca_bid:.4f}",
            f"{alpaca_ask:.4f}",
            f"{alpaca_iv:.4f}",
            f"{alpaca_delta:.4f}",
            source,
            action,
        ]

        with _lock:
            _ensure_file()
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    except Exception as e:
        # Never crash the engine for logging
        pass


def log_option_pricing_event(
    engine: str,
    ticker: str,
    spot: float,
    strike: float,
    expiry,
    option_type: str,
    iv_used: float,
    bs_estimate: float,
    real_quote: dict = None,
    action: str = "entry",
):
    """
    Convenience wrapper — pass the real_quote dict from
    alpaca_options_pricing.get_real_premium_or_fallback().
    """
    rq = real_quote or {}
    log_option_price(
        engine=engine,
        ticker=ticker,
        spot=spot,
        strike=strike,
        expiry=str(expiry)[:10],
        option_type=option_type,
        iv_used=iv_used,
        bs_price=bs_estimate,
        alpaca_mid=rq.get('premium', 0),
        alpaca_bid=rq.get('bid', 0),
        alpaca_ask=rq.get('ask', 0),
        alpaca_iv=rq.get('iv', 0),
        alpaca_delta=rq.get('delta', 0),
        source=rq.get('source', 'bs_only'),
        action=action,
    )
