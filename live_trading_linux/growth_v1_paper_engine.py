"""
Growth v1 Paper Engine — SPY 200-day SMA regime filter on QQQ.

Strategy:
  - Buy QQQ when SPY closes above its 200-day SMA (risk-on).
  - Sell QQQ when SPY closes below its 200-day SMA (risk-off), hold cash.
  - 3-day confirmation required for UPGRADES (risk-off -> risk-on).
  - Immediate exit for DOWNGRADES (risk-on -> risk-off).
  - Cash earns 5% annualized risk-free rate.
  - Also tracks a trailing-stop variant: exit if QQQ drops 8% from peak.

Backtest (19.5 yr): CAGR 14.6%, Sharpe 0.68, MaxDD -20.5%.

Expected frequency: ~4-8 trades per year.
Runs once daily at 16:00 ET via PM2 cron.

Outputs:
  - live_trading_linux/growth_v1_state/state.json
  - live_trading_linux/growth_v1_state/trades.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_DIR = Path(__file__).resolve().parent / "growth_v1_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMA_PERIOD = 200
CONFIRM_DAYS_UPGRADE = 3       # days SPY must stay above SMA to go risk-on
INITIAL_NAV = 100_000.0
RISK_FREE_RATE = 0.05          # 5% annual for cash
TRAILING_STOP_PCT = 0.08       # 8% drawdown from peak triggers exit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GrowthV1] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "growth_v1.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("growth_v1")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "position": None,
        "regime": "risk_off",
        "regime_confirm_days": 0,
        "last_check": None,
        "trailing_stop_variant": {
            "nav": INITIAL_NAV,
            "cash": INITIAL_NAV,
            "position": None,
            "regime": "risk_off",
            "regime_confirm_days": 0,
            "exited_trailing_stop": False,
        },
        "trades": [],
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return default_state()


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def save_trades(trades: list) -> None:
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_market_data() -> dict:
    """Fetch SPY and QQQ data via yfinance."""
    import yfinance as yf

    spy = yf.download("SPY", period="2y", interval="1d", progress=False)
    qqq = yf.download("QQQ", period="5d", interval="1d", progress=False)

    spy_close = spy["Close"].squeeze()
    spy_sma200 = spy_close.rolling(SMA_PERIOD).mean()

    return {
        "spy_close": float(spy_close.iloc[-1]),
        "spy_sma200": float(spy_sma200.iloc[-1]),
        "spy_above_sma": bool(spy_close.iloc[-1] > spy_sma200.iloc[-1]),
        "qqq_close": float(qqq["Close"].squeeze().iloc[-1]),
        "date": spy.index[-1].strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def process_day(state: dict, mkt: dict) -> dict:
    """Run one daily check. Modifies state in place."""
    today = mkt["date"]

    # Skip if already processed today
    if state["last_check"] == today:
        log.info(f"Already processed {today}, skipping.")
        return state

    spy_above = mkt["spy_above_sma"]
    qqq_price = mkt["qqq_close"]
    old_regime = state["regime"]

    # --- Regime transition logic ---
    if old_regime == "risk_off":
        if spy_above:
            state["regime_confirm_days"] += 1
            log.info(
                f"SPY above SMA200 — confirm day "
                f"{state['regime_confirm_days']}/{CONFIRM_DAYS_UPGRADE}"
            )
            if state["regime_confirm_days"] >= CONFIRM_DAYS_UPGRADE:
                # UPGRADE: risk_off -> risk_on
                state["regime"] = "risk_on"
                state["regime_confirm_days"] = 0
                log.info("REGIME UPGRADE: risk_off -> risk_on")
                _enter_position(state, qqq_price, today)
        else:
            # Reset confirmation counter
            if state["regime_confirm_days"] > 0:
                log.info("SPY back below SMA200 — resetting confirm counter.")
            state["regime_confirm_days"] = 0

    elif old_regime == "risk_on":
        if not spy_above:
            # IMMEDIATE DOWNGRADE: risk_on -> risk_off
            state["regime"] = "risk_off"
            state["regime_confirm_days"] = 0
            log.info("REGIME DOWNGRADE: risk_on -> risk_off (immediate)")
            _exit_position(state, qqq_price, today, reason="regime_downgrade")
        else:
            # Still risk-on — update position mark-to-market
            _mark_to_market(state, qqq_price)

    # --- Accrue cash interest ---
    if state["cash"] > 0 and state["position"] is None:
        daily_rate = RISK_FREE_RATE / 252
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = state["cash"]

    # --- Process trailing stop variant ---
    _process_trailing_stop_variant(state, mkt)

    state["last_check"] = today
    return state


def _enter_position(state: dict, price: float, date_str: str) -> None:
    """Buy QQQ with all available cash."""
    shares = int(state["cash"] / price)  # whole shares only
    if shares <= 0:
        log.warning(f"Not enough cash ({state['cash']:.2f}) to buy QQQ at {price:.2f}")
        return

    cost = shares * price
    state["cash"] -= cost
    state["position"] = {
        "ticker": "QQQ",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "peak_price": price,
    }
    state["nav"] = state["cash"] + shares * price

    trade = {
        "action": "BUY",
        "ticker": "QQQ",
        "shares": shares,
        "price": price,
        "date": date_str,
        "nav_after": state["nav"],
        "reason": "regime_upgrade",
    }
    state["trades"].append(trade)
    log.info(
        f"BUY {shares} QQQ @ {price:.2f} = ${cost:,.2f} | "
        f"NAV: ${state['nav']:,.2f}"
    )


def _exit_position(state: dict, price: float, date_str: str, reason: str) -> None:
    """Sell entire QQQ position."""
    pos = state["position"]
    if pos is None:
        return

    proceeds = pos["shares"] * price
    pnl = (price - pos["entry_price"]) * pos["shares"]
    pnl_pct = (price / pos["entry_price"] - 1) * 100

    state["cash"] += proceeds
    state["nav"] = state["cash"]

    trade = {
        "action": "SELL",
        "ticker": "QQQ",
        "shares": pos["shares"],
        "price": price,
        "date": date_str,
        "entry_price": pos["entry_price"],
        "entry_date": pos["entry_date"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "nav_after": state["nav"],
        "reason": reason,
    }
    state["trades"].append(trade)
    state["position"] = None

    log.info(
        f"SELL {pos['shares']} QQQ @ {price:.2f} | "
        f"PnL: ${pnl:,.2f} ({pnl_pct:+.1f}%) | "
        f"NAV: ${state['nav']:,.2f} | Reason: {reason}"
    )


def _mark_to_market(state: dict, price: float) -> None:
    """Update NAV and peak price for current position."""
    pos = state["position"]
    if pos is None:
        return

    pos["peak_price"] = max(pos["peak_price"], price)
    state["nav"] = state["cash"] + pos["shares"] * price


def _process_trailing_stop_variant(state: dict, mkt: dict) -> None:
    """Mirror logic for the trailing-stop variant."""
    ts = state["trailing_stop_variant"]
    spy_above = mkt["spy_above_sma"]
    qqq_price = mkt["qqq_close"]
    today = mkt["date"]
    old_regime = ts["regime"]

    # Reset trailing stop flag on new regime cycle
    if old_regime == "risk_off":
        ts["exited_trailing_stop"] = False
        if spy_above:
            ts["regime_confirm_days"] += 1
            if ts["regime_confirm_days"] >= CONFIRM_DAYS_UPGRADE:
                ts["regime"] = "risk_on"
                ts["regime_confirm_days"] = 0
                _ts_enter(ts, qqq_price, today)
        else:
            ts["regime_confirm_days"] = 0

    elif old_regime == "risk_on":
        if not spy_above:
            ts["regime"] = "risk_off"
            ts["regime_confirm_days"] = 0
            if ts["position"] is not None:
                _ts_exit(ts, qqq_price, today, "regime_downgrade")
        elif ts["position"] is not None:
            # Check trailing stop
            ts["position"]["peak_price"] = max(
                ts["position"]["peak_price"], qqq_price
            )
            drawdown = 1 - qqq_price / ts["position"]["peak_price"]
            if drawdown >= TRAILING_STOP_PCT:
                log.info(
                    f"[TS-VARIANT] Trailing stop hit: "
                    f"{drawdown*100:.1f}% from peak {ts['position']['peak_price']:.2f}"
                )
                _ts_exit(ts, qqq_price, today, "trailing_stop_8pct")
                ts["exited_trailing_stop"] = True
            else:
                # Mark to market
                ts["nav"] = ts["cash"] + ts["position"]["shares"] * qqq_price

    # Cash interest
    if ts["cash"] > 0 and ts["position"] is None:
        daily_rate = RISK_FREE_RATE / 252
        ts["cash"] += ts["cash"] * daily_rate
        ts["nav"] = ts["cash"]


def _ts_enter(ts: dict, price: float, date_str: str) -> None:
    shares = int(ts["cash"] / price)
    if shares <= 0:
        return
    cost = shares * price
    ts["cash"] -= cost
    ts["position"] = {
        "ticker": "QQQ",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "peak_price": price,
    }
    ts["nav"] = ts["cash"] + shares * price
    log.info(
        f"[TS-VARIANT] BUY {shares} QQQ @ {price:.2f} | NAV: ${ts['nav']:,.2f}"
    )


def _ts_exit(ts: dict, price: float, date_str: str, reason: str) -> None:
    pos = ts["position"]
    if pos is None:
        return
    proceeds = pos["shares"] * price
    pnl = (price - pos["entry_price"]) * pos["shares"]
    ts["cash"] += proceeds
    ts["nav"] = ts["cash"]
    ts["position"] = None
    log.info(
        f"[TS-VARIANT] SELL @ {price:.2f} | "
        f"PnL: ${pnl:,.2f} | NAV: ${ts['nav']:,.2f} | Reason: {reason}"
    )


# ---------------------------------------------------------------------------
# Initialization — determine starting state from market data
# ---------------------------------------------------------------------------
def initialize_state() -> dict:
    """Create initial state based on current market conditions."""
    import yfinance as yf

    log.info("Initializing Growth v1 state from current market data...")

    spy = yf.download("SPY", period="2y", interval="1d", progress=False)
    qqq = yf.download("QQQ", period="5d", interval="1d", progress=False)

    spy_close = spy["Close"].squeeze()
    spy_sma200 = spy_close.rolling(SMA_PERIOD).mean()

    # Check how many recent days SPY has been above SMA200
    days_above = 0
    for i in range(-1, -11, -1):
        if spy_close.iloc[i] > spy_sma200.iloc[i]:
            days_above += 1
        else:
            break

    qqq_price = float(qqq["Close"].squeeze().iloc[-1])
    today = spy.index[-1].strftime("%Y-%m-%d")
    spy_price = float(spy_close.iloc[-1])
    sma_val = float(spy_sma200.iloc[-1])

    log.info(
        f"SPY: {spy_price:.2f} | SMA200: {sma_val:.2f} | "
        f"Above: {spy_price > sma_val} | Consecutive days above: {days_above}"
    )

    state = default_state()

    # If SPY has been above SMA200 for 3+ days, start risk-on with position
    if days_above >= CONFIRM_DAYS_UPGRADE:
        state["regime"] = "risk_on"
        state["regime_confirm_days"] = 0

        # Enter QQQ position at current price
        shares = int(state["cash"] / qqq_price)
        cost = shares * qqq_price
        state["cash"] -= cost
        state["position"] = {
            "ticker": "QQQ",
            "shares": shares,
            "entry_price": qqq_price,
            "entry_date": today,
            "peak_price": qqq_price,
        }
        state["nav"] = state["cash"] + shares * qqq_price

        trade = {
            "action": "BUY",
            "ticker": "QQQ",
            "shares": shares,
            "price": qqq_price,
            "date": today,
            "nav_after": state["nav"],
            "reason": "initialization_risk_on",
        }
        state["trades"].append(trade)

        # Mirror for trailing stop variant
        ts = state["trailing_stop_variant"]
        ts["regime"] = "risk_on"
        ts["regime_confirm_days"] = 0
        ts_shares = int(ts["cash"] / qqq_price)
        ts_cost = ts_shares * qqq_price
        ts["cash"] -= ts_cost
        ts["position"] = {
            "ticker": "QQQ",
            "shares": ts_shares,
            "entry_price": qqq_price,
            "entry_date": today,
            "peak_price": qqq_price,
        }
        ts["nav"] = ts["cash"] + ts_shares * qqq_price

        log.info(
            f"Initialized RISK-ON: {shares} QQQ @ {qqq_price:.2f} | "
            f"NAV: ${state['nav']:,.2f}"
        )
    else:
        state["regime_confirm_days"] = days_above
        log.info(
            f"Initialized RISK-OFF (only {days_above} days above SMA200, "
            f"need {CONFIRM_DAYS_UPGRADE})"
        )

    state["last_check"] = today
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Growth v1 Paper Engine — daily check")
    log.info("=" * 60)

    # Load or initialize state
    if STATE_FILE.exists():
        state = load_state()
        log.info(
            f"Loaded state: regime={state['regime']}, "
            f"NAV=${state['nav']:,.2f}, "
            f"position={'QQQ' if state['position'] else 'CASH'}"
        )
    else:
        state = initialize_state()
        save_state(state)
        save_trades(state["trades"])
        log.info("Initial state saved.")
        return

    # Fetch market data
    try:
        mkt = get_market_data()
    except Exception as e:
        log.error(f"Failed to fetch market data: {e}")
        return

    log.info(
        f"Market: SPY={mkt['spy_close']:.2f} SMA200={mkt['spy_sma200']:.2f} "
        f"{'ABOVE' if mkt['spy_above_sma'] else 'BELOW'} | "
        f"QQQ={mkt['qqq_close']:.2f} | Date={mkt['date']}"
    )

    # Process
    state = process_day(state, mkt)

    # Save
    save_state(state)
    save_trades(state["trades"])

    # Summary
    ts = state["trailing_stop_variant"]
    log.info(
        f"END: Regime={state['regime']} | "
        f"NAV=${state['nav']:,.2f} | "
        f"Position={'QQQ' if state['position'] else 'CASH'} | "
        f"TS-Variant NAV=${ts['nav']:,.2f} "
        f"({'QQQ' if ts['position'] else 'CASH'})"
    )
    log.info(f"Total trades: {len(state['trades'])}")


if __name__ == "__main__":
    main()
