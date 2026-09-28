"""
BTC Trend Paper Engine — 100-day SMA trend-following on BTC-USD.

Strategy (winner from growth research, 34% standalone CAGR):
  - Hold BTC when price > 100-day SMA, go to cash when below.
  - Vol target: 50% annualized. Scale position by target_vol / BTC_20d_realized_vol,
    capped at 1.0x (no leverage).
  - Cash earns 5% annualized risk-free rate.

Designed to run daily at 16:15 ET via PM2 cron.
Idempotent: safe to run twice on the same day (detects and skips).

Outputs:
  - data/paper_engines/btc_trend/state.json
  - data/paper_engines/btc_trend/trades.json
  - data/paper_engines/btc_trend/daily_returns.csv
  - logs/btc_trend_paper.log
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # /home/jupiter/Lvl3Quant
STATE_DIR = ROOT / "data" / "paper_engines" / "btc_trend"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"
DAILY_RETURNS_FILE = STATE_DIR / "daily_returns.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMA_WINDOW = 100                # 100-day simple moving average
VOL_TARGET = 0.50               # 50% annualized target vol (BTC is volatile)
REALIZED_VOL_WINDOW = 20        # 20 days for realized vol
POSITION_CAP = 1.0              # Max 1.0x NAV (no leverage)
INITIAL_NAV = 100_000.0
RISK_FREE_RATE = 0.05           # 5% annual for cash

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BTCTrend] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "btc_trend_paper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("btc_trend")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "position": None,       # None = cash, or {"ticker": "BTC-USD", "units": N, ...}
        "signal": "CASH",       # "BTC" or "CASH"
        "last_check": None,
        "last_nav": INITIAL_NAV,
        "inception_date": today_str,
        "peak_nav": INITIAL_NAV,
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


def append_daily_return(date_str: str, nav: float, daily_ret: float, signal: str,
                        btc_price: float, sma_100: float, vol_scale: float) -> None:
    """Append one row to daily_returns.csv."""
    file_exists = DAILY_RETURNS_FILE.exists()
    with open(DAILY_RETURNS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "nav", "daily_return", "signal", "btc_price", "sma_100", "vol_scale"])
        writer.writerow([date_str, f"{nav:.2f}", f"{daily_ret:.6f}", signal, f"{btc_price:.2f}",
                         f"{sma_100:.2f}", f"{vol_scale:.4f}"])


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_market_data() -> dict:
    """Fetch BTC-USD data via yfinance."""
    import yfinance as yf

    # Need enough history for 100-day SMA + 20-day realized vol
    btc_df = yf.download("BTC-USD", period="6mo", interval="1d", progress=False)

    if len(btc_df) < SMA_WINDOW:
        raise ValueError(f"Not enough BTC data: got {len(btc_df)} days, need {SMA_WINDOW}")

    btc_close = btc_df["Close"].squeeze()

    # Current price
    btc_price = float(btc_close.iloc[-1])

    # 100-day SMA
    sma_100 = float(btc_close.rolling(SMA_WINDOW).mean().iloc[-1])

    # 20-day realized vol (annualized, using 365 days for crypto)
    btc_returns = btc_close.pct_change().dropna()
    btc_realized_vol = float(btc_returns.tail(REALIZED_VOL_WINDOW).std() * np.sqrt(365)) \
        if len(btc_returns) >= REALIZED_VOL_WINDOW else 0.60

    trade_date = btc_df.index[-1].strftime("%Y-%m-%d")

    return {
        "btc_price": btc_price,
        "sma_100": sma_100,
        "btc_realized_vol": btc_realized_vol,
        "date": trade_date,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def process_day(state: dict, mkt: dict) -> dict:
    """Run one daily check. Modifies state in place and returns it."""
    today = mkt["date"]

    # Idempotency: skip if already processed today
    if state["last_check"] == today:
        log.info(f"Already processed {today}, skipping.")
        return state

    btc_price = mkt["btc_price"]
    sma_100 = mkt["sma_100"]
    btc_realized_vol = mkt["btc_realized_vol"]
    prev_nav = state["nav"]

    # --- Determine signal ---
    if btc_price > sma_100:
        signal = "BTC"
        signal_reason = f"BTC ${btc_price:,.0f} > SMA100 ${sma_100:,.0f} (trend UP)"
    else:
        signal = "CASH"
        signal_reason = f"BTC ${btc_price:,.0f} <= SMA100 ${sma_100:,.0f} (trend DOWN)"

    log.info(f"Signal: {signal} | {signal_reason}")

    # --- Compute vol-target position scale ---
    vol_scale = min(VOL_TARGET / max(btc_realized_vol, 0.05), POSITION_CAP)
    log.info(f"Vol-target: {VOL_TARGET:.0%} / {btc_realized_vol:.1%} = {vol_scale:.2%} scale (capped at {POSITION_CAP:.0%})")

    # --- Execute transitions ---
    old_signal = state["signal"]

    if signal == "BTC" and old_signal == "CASH":
        # Transition: CASH -> BTC
        _enter_position(state, btc_price, today, vol_scale)

    elif signal == "CASH" and old_signal == "BTC":
        # Transition: BTC -> CASH
        _exit_position(state, btc_price, today, signal_reason)

    elif signal == "BTC" and old_signal == "BTC":
        # Stay in BTC -- mark to market and check if rebalance needed
        _mark_to_market(state, btc_price)
        _maybe_rebalance(state, btc_price, vol_scale, today)

    elif signal == "CASH" and old_signal == "CASH":
        # Stay in cash -- accrue interest
        pass

    # --- Accrue cash interest (on any uninvested cash) ---
    if state["position"] is None:
        daily_rate = RISK_FREE_RATE / 365  # crypto runs 365 days
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = state["cash"]
    else:
        # Also accrue on residual cash from partial position
        if state["cash"] > 1.0:
            daily_rate = RISK_FREE_RATE / 365
            state["cash"] += state["cash"] * daily_rate
            state["nav"] = state["cash"] + state["position"]["units"] * btc_price

    state["signal"] = signal
    state["last_check"] = today

    # Track peak NAV for drawdown
    state["peak_nav"] = max(state.get("peak_nav", state["nav"]), state["nav"])

    # --- Record daily return ---
    daily_ret = (state["nav"] / prev_nav - 1) if prev_nav > 0 else 0.0
    state["last_nav"] = state["nav"]
    append_daily_return(today, state["nav"], daily_ret, signal, btc_price, sma_100, vol_scale)

    return state


def _enter_position(state: dict, price: float, date_str: str, vol_scale: float) -> None:
    """Buy BTC-USD with vol-target-scaled position size."""
    investable = state["cash"] * vol_scale
    units = investable / price  # fractional BTC allowed
    if units <= 0:
        log.warning(f"Not enough cash ({state['cash']:.2f}) or vol_scale too low ({vol_scale:.2%}) to buy BTC at ${price:,.2f}")
        return

    cost = units * price
    state["cash"] -= cost
    state["position"] = {
        "ticker": "BTC-USD",
        "units": units,
        "entry_price": price,
        "entry_date": date_str,
        "vol_scale": vol_scale,
    }
    state["nav"] = state["cash"] + units * price
    state["signal"] = "BTC"

    trade = {
        "action": "BUY",
        "ticker": "BTC-USD",
        "units": round(units, 8),
        "price": round(price, 2),
        "date": date_str,
        "nav_after": round(state["nav"], 2),
        "vol_scale": round(vol_scale, 4),
        "reason": "signal_btc",
    }
    state["trades"].append(trade)
    log.info(f"BUY {units:.6f} BTC @ ${price:,.2f} (vol_scale={vol_scale:.0%}) | NAV: ${state['nav']:,.2f}")


def _exit_position(state: dict, price: float, date_str: str, reason: str) -> None:
    """Sell entire BTC position."""
    pos = state["position"]
    if pos is None:
        return

    proceeds = pos["units"] * price
    pnl = (price - pos["entry_price"]) * pos["units"]
    pnl_pct = (price / pos["entry_price"] - 1) * 100

    state["cash"] += proceeds
    state["nav"] = state["cash"]
    state["signal"] = "CASH"

    trade = {
        "action": "SELL",
        "ticker": "BTC-USD",
        "units": round(pos["units"], 8),
        "price": round(price, 2),
        "date": date_str,
        "entry_price": round(pos["entry_price"], 2),
        "entry_date": pos["entry_date"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "nav_after": round(state["nav"], 2),
        "reason": reason,
    }
    state["trades"].append(trade)
    state["position"] = None

    log.info(
        f"SELL {pos['units']:.6f} BTC @ ${price:,.2f} | "
        f"PnL: ${pnl:,.2f} ({pnl_pct:+.1f}%) | "
        f"NAV: ${state['nav']:,.2f} | Reason: {reason}"
    )


def _mark_to_market(state: dict, price: float) -> None:
    """Update NAV for current BTC position."""
    pos = state["position"]
    if pos is None:
        return
    state["nav"] = state["cash"] + pos["units"] * price


def _maybe_rebalance(state: dict, price: float, vol_scale: float, date_str: str) -> None:
    """Rebalance position if vol_scale has changed significantly (>20% delta)."""
    pos = state["position"]
    if pos is None:
        return

    current_units = pos["units"]
    total_value = state["cash"] + current_units * price
    target_investable = total_value * vol_scale
    target_units = target_investable / price

    if current_units <= 0:
        return

    unit_delta_pct = abs(target_units - current_units) / current_units
    if unit_delta_pct > 0.20:
        delta = target_units - current_units
        if delta > 0:
            # Buy more
            buy_cost = delta * price
            if buy_cost <= state["cash"]:
                state["cash"] -= buy_cost
                pos["units"] = target_units
                pos["vol_scale"] = vol_scale
                log.info(f"REBALANCE +{delta:.6f} BTC @ ${price:,.2f} (vol_scale={vol_scale:.0%})")
                trade = {
                    "action": "REBALANCE_BUY",
                    "ticker": "BTC-USD",
                    "units": round(delta, 8),
                    "price": round(price, 2),
                    "date": date_str,
                    "vol_scale": round(vol_scale, 4),
                    "reason": "vol_rebalance",
                }
                state["trades"].append(trade)
        elif delta < 0:
            # Sell some
            sell_proceeds = abs(delta) * price
            state["cash"] += sell_proceeds
            pos["units"] = target_units
            pos["vol_scale"] = vol_scale
            log.info(f"REBALANCE {delta:.6f} BTC @ ${price:,.2f} (vol_scale={vol_scale:.0%})")
            trade = {
                "action": "REBALANCE_SELL",
                "ticker": "BTC-USD",
                "units": round(abs(delta), 8),
                "price": round(price, 2),
                "date": date_str,
                "vol_scale": round(vol_scale, 4),
                "reason": "vol_rebalance",
            }
            state["trades"].append(trade)

        state["nav"] = state["cash"] + pos["units"] * price


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("BTC Trend Paper Engine — daily check")
    log.info("=" * 60)

    # Load state
    if STATE_FILE.exists():
        state = load_state()
        log.info(
            f"Loaded state: signal={state['signal']}, "
            f"NAV=${state['nav']:,.2f}, "
            f"position={'BTC' if state['position'] else 'CASH'}"
        )
    else:
        log.info("No state file found — creating initial state.")
        state = default_state()
        save_state(state)
        save_trades([])
        log.info(f"Initial state saved: NAV=${state['nav']:,.2f} (CASH)")
        # On first run, still try to process today's signal
        # Fall through to market data fetch

    # Fetch market data
    try:
        mkt = get_market_data()
    except Exception as e:
        log.error(f"Failed to fetch market data: {e}")
        log.warning("Staying in current position due to data failure.")
        print(f"[BTCTrend] ERROR: data fetch failed — {e}")
        return

    log.info(
        f"Market: BTC=${mkt['btc_price']:,.2f} | "
        f"SMA100=${mkt['sma_100']:,.2f} | "
        f"RealVol={mkt['btc_realized_vol']:.1%} | "
        f"Date={mkt['date']}"
    )

    # Process
    state = process_day(state, mkt)

    # Save
    save_state(state)
    save_trades(state["trades"])

    # Compute drawdown
    peak = state.get("peak_nav", state["nav"])
    dd = (state["nav"] / peak - 1) * 100 if peak > 0 else 0.0

    # One-line PM2 summary
    if state["position"]:
        pos_str = f"BTC x{state['position']['units']:.6f}"
    else:
        pos_str = "CASH"
    summary = (
        f"[BTCTrend] {mkt['date']} | "
        f"Signal={state['signal']} | {pos_str} | "
        f"NAV=${state['nav']:,.2f} | "
        f"DD={dd:.1f}% | "
        f"BTC=${mkt['btc_price']:,.0f} SMA100=${mkt['sma_100']:,.0f} | "
        f"Trades={len(state['trades'])}"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
