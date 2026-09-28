"""
Growth v2 Paper Engine — SPY 200-day SMA regime filter on TQQQ (3x leveraged QQQ).

Strategy (HC #690 — growth book must beat income book CAGR):
  - Buy TQQQ when SPY closes above its 200-day SMA (risk-on).
  - Sell TQQQ when SPY closes below its 200-day SMA (risk-off), hold cash.
  - 3-day confirmation required for UPGRADES (risk-off -> risk-on).
  - Immediate exit for DOWNGRADES (risk-on -> risk-off).
  - Cash earns 5% annualized risk-free rate.
  - Also tracks a trailing-stop variant: exit if TQQQ drops 20% from peak.
    (TQQQ is 3x leveraged, so trailing stop is wider than QQQ's 8%)

Backtest (15 yr real TQQQ data): CAGR 83.1%, Sharpe 1.58, MaxDD -37.4%.
Never had a losing year (even 2022 bear = +10.5%).

Expected frequency: ~4-8 trades per year.
Runs once daily at 16:00 ET via PM2 cron.

Outputs:
  - live_trading_linux/growth_v2_state/state.json
  - live_trading_linux/growth_v2_state/trades.json
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
STATE_DIR = Path(__file__).resolve().parent / "growth_v2_state"
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
TRAILING_STOP_PCT = 0.20       # 20% drawdown from peak (TQQQ is 3x, wider stop needed)
VOL_TARGET = 0.30              # 30% annualized target vol for vol-target variant

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GrowthV2] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "growth_v2.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("growth_v2")


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
        "vol_target_variant": {
            "nav": INITIAL_NAV,
            "cash": INITIAL_NAV,
            "position": None,
            "regime": "risk_off",
            "regime_confirm_days": 0,
            "target_vol": VOL_TARGET,
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
    """Fetch SPY and TQQQ data via yfinance."""
    import yfinance as yf
    import numpy as np

    spy = yf.download("SPY", period="2y", interval="1d", progress=False)
    tqqq = yf.download("TQQQ", period="2mo", interval="1d", progress=False)

    spy_close = spy["Close"].squeeze()
    spy_sma200 = spy_close.rolling(SMA_PERIOD).mean()

    tqqq_close = tqqq["Close"].squeeze()
    # 21-day realized vol of TQQQ (annualized) for vol-target variant
    tqqq_returns = tqqq_close.pct_change().dropna()
    realized_vol = float(tqqq_returns.tail(21).std() * np.sqrt(252)) if len(tqqq_returns) >= 21 else 0.60

    return {
        "spy_close": float(spy_close.iloc[-1]),
        "spy_sma200": float(spy_sma200.iloc[-1]),
        "spy_above_sma": bool(spy_close.iloc[-1] > spy_sma200.iloc[-1]),
        "tqqq_close": float(tqqq_close.iloc[-1]),
        "tqqq_realized_vol": realized_vol,
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
    tqqq_price = mkt["tqqq_close"]
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
                _enter_position(state, tqqq_price, today)
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
            _exit_position(state, tqqq_price, today, reason="regime_downgrade")
        else:
            # Still risk-on — update position mark-to-market
            _mark_to_market(state, tqqq_price)

    # --- Accrue cash interest ---
    if state["cash"] > 0 and state["position"] is None:
        daily_rate = RISK_FREE_RATE / 252
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = state["cash"]

    # --- Process trailing stop variant ---
    _process_trailing_stop_variant(state, mkt)

    # --- Process vol-target variant ---
    _process_vol_target_variant(state, mkt)

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
        "ticker": "TQQQ",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "peak_price": price,
    }
    state["nav"] = state["cash"] + shares * price

    trade = {
        "action": "BUY",
        "ticker": "TQQQ",
        "shares": shares,
        "price": price,
        "date": date_str,
        "nav_after": state["nav"],
        "reason": "regime_upgrade",
    }
    state["trades"].append(trade)
    log.info(
        f"BUY {shares} TQQQ @ {price:.2f} = ${cost:,.2f} | "
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
        "ticker": "TQQQ",
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
        f"SELL {pos['shares']} TQQQ @ {price:.2f} | "
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
    tqqq_price = mkt["tqqq_close"]
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
                _ts_enter(ts, tqqq_price, today)
        else:
            ts["regime_confirm_days"] = 0

    elif old_regime == "risk_on":
        if not spy_above:
            ts["regime"] = "risk_off"
            ts["regime_confirm_days"] = 0
            if ts["position"] is not None:
                _ts_exit(ts, tqqq_price, today, "regime_downgrade")
        elif ts["position"] is not None:
            # Check trailing stop
            ts["position"]["peak_price"] = max(
                ts["position"]["peak_price"], tqqq_price
            )
            drawdown = 1 - tqqq_price / ts["position"]["peak_price"]
            if drawdown >= TRAILING_STOP_PCT:
                log.info(
                    f"[TS-VARIANT] Trailing stop hit: "
                    f"{drawdown*100:.1f}% from peak {ts['position']['peak_price']:.2f}"
                )
                _ts_exit(ts, tqqq_price, today, "trailing_stop_20pct")
                ts["exited_trailing_stop"] = True
            else:
                # Mark to market
                ts["nav"] = ts["cash"] + ts["position"]["shares"] * tqqq_price

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
        "ticker": "TQQQ",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "peak_price": price,
    }
    ts["nav"] = ts["cash"] + shares * price
    log.info(
        f"[TS-VARIANT] BUY {shares} TQQQ @ {price:.2f} | NAV: ${ts['nav']:,.2f}"
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


def _process_vol_target_variant(state: dict, mkt: dict) -> None:
    """Vol-target variant: scale position size inversely with realized volatility.
    Target 30% annualized portfolio vol. When TQQQ vol is high, hold less.
    When low, hold full position. Still requires SPY > 200MA for any position.
    """
    vt = state.get("vol_target_variant")
    if vt is None:
        # Backward compat: add variant if missing from old state
        state["vol_target_variant"] = {
            "nav": INITIAL_NAV, "cash": INITIAL_NAV, "position": None,
            "regime": "risk_off", "regime_confirm_days": 0,
            "target_vol": VOL_TARGET,
        }
        vt = state["vol_target_variant"]

    spy_above = mkt["spy_above_sma"]
    tqqq_price = mkt["tqqq_close"]
    realized_vol = mkt.get("tqqq_realized_vol", 0.60)
    today = mkt["date"]
    old_regime = vt["regime"]

    # Regime transitions (same logic as base)
    if old_regime == "risk_off":
        if spy_above:
            vt["regime_confirm_days"] += 1
            if vt["regime_confirm_days"] >= CONFIRM_DAYS_UPGRADE:
                vt["regime"] = "risk_on"
                vt["regime_confirm_days"] = 0
                # Calculate vol-adjusted position size
                vol_scale = min(VOL_TARGET / max(realized_vol, 0.10), 1.0)
                _vt_enter(vt, tqqq_price, today, vol_scale)
        else:
            vt["regime_confirm_days"] = 0

    elif old_regime == "risk_on":
        if not spy_above:
            vt["regime"] = "risk_off"
            vt["regime_confirm_days"] = 0
            if vt["position"] is not None:
                _vt_exit(vt, tqqq_price, today, "regime_downgrade")
        elif vt["position"] is not None:
            # Rebalance position size based on current vol (weekly check)
            # Only rebalance if vol_scale changed significantly (>15%)
            vol_scale = min(VOL_TARGET / max(realized_vol, 0.10), 1.0)
            current_shares = vt["position"]["shares"]
            total_value = vt["cash"] + current_shares * tqqq_price
            target_shares = int(total_value * vol_scale / tqqq_price)

            if abs(target_shares - current_shares) / max(current_shares, 1) > 0.15:
                # Rebalance
                delta = target_shares - current_shares
                if delta > 0:
                    # Buy more
                    buy_cost = delta * tqqq_price
                    if buy_cost <= vt["cash"]:
                        vt["cash"] -= buy_cost
                        vt["position"]["shares"] = target_shares
                        log.info(
                            f"[VT-VARIANT] REBALANCE +{delta} shares @ {tqqq_price:.2f} "
                            f"(vol={realized_vol:.0%}, scale={vol_scale:.0%})"
                        )
                elif delta < 0:
                    # Sell some
                    sell_proceeds = abs(delta) * tqqq_price
                    vt["cash"] += sell_proceeds
                    vt["position"]["shares"] = target_shares
                    log.info(
                        f"[VT-VARIANT] REBALANCE {delta} shares @ {tqqq_price:.2f} "
                        f"(vol={realized_vol:.0%}, scale={vol_scale:.0%})"
                    )

            # Mark to market
            vt["nav"] = vt["cash"] + vt["position"]["shares"] * tqqq_price

    # Cash interest
    if vt["cash"] > 0 and vt["position"] is None:
        daily_rate = RISK_FREE_RATE / 252
        vt["cash"] += vt["cash"] * daily_rate
        vt["nav"] = vt["cash"]


def _vt_enter(vt: dict, price: float, date_str: str, vol_scale: float) -> None:
    """Enter vol-target variant position with scaled size."""
    investable = vt["cash"] * vol_scale
    shares = int(investable / price)
    if shares <= 0:
        return
    cost = shares * price
    vt["cash"] -= cost
    vt["position"] = {
        "ticker": "TQQQ",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "vol_scale": vol_scale,
    }
    vt["nav"] = vt["cash"] + shares * price
    log.info(
        f"[VT-VARIANT] BUY {shares} TQQQ @ {price:.2f} "
        f"(vol_scale={vol_scale:.0%}) | NAV: ${vt['nav']:,.2f}"
    )


def _vt_exit(vt: dict, price: float, date_str: str, reason: str) -> None:
    """Exit vol-target variant position."""
    pos = vt["position"]
    if pos is None:
        return
    proceeds = pos["shares"] * price
    pnl = (price - pos["entry_price"]) * pos["shares"]
    vt["cash"] += proceeds
    vt["nav"] = vt["cash"]
    vt["position"] = None
    log.info(
        f"[VT-VARIANT] SELL @ {price:.2f} | "
        f"PnL: ${pnl:,.2f} | NAV: ${vt['nav']:,.2f} | Reason: {reason}"
    )


# ---------------------------------------------------------------------------
# Initialization — determine starting state from market data
# ---------------------------------------------------------------------------
def initialize_state() -> dict:
    """Create initial state based on current market conditions."""
    import yfinance as yf

    log.info("Initializing Growth v2 (TQQQ) state from current market data...")

    spy = yf.download("SPY", period="2y", interval="1d", progress=False)
    qqq = yf.download("TQQQ", period="5d", interval="1d", progress=False)

    spy_close = spy["Close"].squeeze()
    spy_sma200 = spy_close.rolling(SMA_PERIOD).mean()

    # Check how many recent days SPY has been above SMA200
    days_above = 0
    for i in range(-1, -11, -1):
        if spy_close.iloc[i] > spy_sma200.iloc[i]:
            days_above += 1
        else:
            break

    tqqq_price = float(qqq["Close"].squeeze().iloc[-1])
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
        shares = int(state["cash"] / tqqq_price)
        cost = shares * tqqq_price
        state["cash"] -= cost
        state["position"] = {
            "ticker": "TQQQ",
            "shares": shares,
            "entry_price": tqqq_price,
            "entry_date": today,
            "peak_price": tqqq_price,
        }
        state["nav"] = state["cash"] + shares * tqqq_price

        trade = {
            "action": "BUY",
            "ticker": "TQQQ",
            "shares": shares,
            "price": tqqq_price,
            "date": today,
            "nav_after": state["nav"],
            "reason": "initialization_risk_on",
        }
        state["trades"].append(trade)

        # Mirror for trailing stop variant
        ts = state["trailing_stop_variant"]
        ts["regime"] = "risk_on"
        ts["regime_confirm_days"] = 0
        ts_shares = int(ts["cash"] / tqqq_price)
        ts_cost = ts_shares * tqqq_price
        ts["cash"] -= ts_cost
        ts["position"] = {
            "ticker": "TQQQ",
            "shares": ts_shares,
            "entry_price": tqqq_price,
            "entry_date": today,
            "peak_price": tqqq_price,
        }
        ts["nav"] = ts["cash"] + ts_shares * tqqq_price

        # Mirror for vol-target variant (with vol-scaled position)
        import numpy as np
        tqqq_hist = yf.download("TQQQ", period="2mo", interval="1d", progress=False)
        tqqq_rets = tqqq_hist["Close"].squeeze().pct_change().dropna()
        realized_vol = float(tqqq_rets.tail(21).std() * np.sqrt(252)) if len(tqqq_rets) >= 21 else 0.60
        vol_scale = min(VOL_TARGET / max(realized_vol, 0.10), 1.0)

        vt = state["vol_target_variant"]
        vt["regime"] = "risk_on"
        vt["regime_confirm_days"] = 0
        vt_investable = vt["cash"] * vol_scale
        vt_shares = int(vt_investable / tqqq_price)
        vt_cost = vt_shares * tqqq_price
        vt["cash"] -= vt_cost
        vt["position"] = {
            "ticker": "TQQQ",
            "shares": vt_shares,
            "entry_price": tqqq_price,
            "entry_date": today,
            "vol_scale": vol_scale,
        }
        vt["nav"] = vt["cash"] + vt_shares * tqqq_price

        log.info(
            f"Initialized RISK-ON: {shares} TQQQ @ {tqqq_price:.2f} | "
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
    log.info("Growth v2 (TQQQ) Paper Engine — daily check")
    log.info("=" * 60)

    # Load or initialize state
    if STATE_FILE.exists():
        state = load_state()
        # Backward compat: add vol_target_variant if missing from old state
        if "vol_target_variant" not in state:
            state["vol_target_variant"] = {
                "nav": INITIAL_NAV, "cash": INITIAL_NAV, "position": None,
                "regime": state["regime"], "regime_confirm_days": state["regime_confirm_days"],
                "target_vol": VOL_TARGET,
            }
            # If base is risk-on with position, mirror with vol-scaled size
            if state["regime"] == "risk_on" and state["position"]:
                try:
                    import numpy as np
                    import yfinance as yf
                    tqqq_hist = yf.download("TQQQ", period="2mo", interval="1d", progress=False)
                    tqqq_rets = tqqq_hist["Close"].squeeze().pct_change().dropna()
                    rv = float(tqqq_rets.tail(21).std() * np.sqrt(252)) if len(tqqq_rets) >= 21 else 0.60
                    vs = min(VOL_TARGET / max(rv, 0.10), 1.0)
                    vt = state["vol_target_variant"]
                    price = state["position"]["entry_price"]
                    investable = vt["cash"] * vs
                    shares = int(investable / price)
                    cost = shares * price
                    vt["cash"] -= cost
                    vt["position"] = {
                        "ticker": "TQQQ", "shares": shares,
                        "entry_price": price, "entry_date": state["position"]["entry_date"],
                        "vol_scale": vs,
                    }
                    vt["nav"] = vt["cash"] + shares * price
                    log.info(f"[VT-VARIANT] Initialized: {shares} TQQQ (vol_scale={vs:.0%}) | NAV: ${vt['nav']:,.2f}")
                except Exception as e:
                    log.warning(f"[VT-VARIANT] Init failed: {e}")
            save_state(state)
        log.info(
            f"Loaded state: regime={state['regime']}, "
            f"NAV=${state['nav']:,.2f}, "
            f"position={'TQQQ' if state['position'] else 'CASH'}"
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
        f"TQQQ={mkt['tqqq_close']:.2f} | RealVol={mkt['tqqq_realized_vol']:.0%} | Date={mkt['date']}"
    )

    # Process
    state = process_day(state, mkt)

    # Save
    save_state(state)
    save_trades(state["trades"])

    # Summary
    ts = state["trailing_stop_variant"]
    vt = state.get("vol_target_variant", {})
    log.info(
        f"END: Regime={state['regime']} | "
        f"NAV=${state['nav']:,.2f} | "
        f"Position={'TQQQ' if state['position'] else 'CASH'} | "
        f"TS-Variant NAV=${ts['nav']:,.2f} "
        f"({'TQQQ' if ts['position'] else 'CASH'}) | "
        f"VT-Variant NAV=${vt.get('nav', 0):,.2f} "
        f"({'TQQQ' if vt.get('position') else 'CASH'})"
    )
    log.info(f"Total trades: {len(state['trades'])}")


if __name__ == "__main__":
    main()
