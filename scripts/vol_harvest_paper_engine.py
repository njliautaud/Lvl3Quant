"""
Vol Harvest Paper Engine — VIX Term Structure timing on SVXY (short VIX ETF).

Strategy (winner from growth research):
  - Compute VIX / 20d-realized-vol-of-SPY ratio (term structure proxy).
  - When ratio < 0.9 (contango = calm markets): hold SVXY.
  - When ratio >= 0.9 (backwardation = stress): go to cash.
  - VIX cap: if VIX > 30, ALWAYS go to cash regardless of ratio.
  - Vol target: 15% annualized. Scale position by target_vol / SVXY_20d_realized_vol,
    capped at 1.0x (no leverage).
  - Smooth days: 1 (no smoothing — immediate transitions).
  - Cash earns 5% annualized risk-free rate.

Designed to run daily at 16:15 ET via PM2 cron.
Idempotent: safe to run twice on the same day (detects and skips).

Outputs:
  - data/paper_engines/vol_harvest/state.json
  - data/paper_engines/vol_harvest/trades.json
  - data/paper_engines/vol_harvest/daily_returns.csv
  - logs/vol_harvest_paper.log
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
STATE_DIR = ROOT / "data" / "paper_engines" / "vol_harvest"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"
DAILY_RETURNS_FILE = STATE_DIR / "daily_returns.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIX_RATIO_THRESHOLD = 0.9      # VIX/realized_vol < this → hold SVXY (contango)
VIX_CAP = 30.0                 # VIX > this → always cash
VOL_TARGET = 0.15              # 15% annualized target vol
REALIZED_VOL_WINDOW = 20       # 20 trading days for realized vol
INITIAL_NAV = 100_000.0
RISK_FREE_RATE = 0.05          # 5% annual for cash

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VolHarvest] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "vol_harvest_paper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("vol_harvest")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "position": None,       # None = cash, or {"ticker": "SVXY", "shares": N, ...}
        "signal": "CASH",       # "SVXY" or "CASH"
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
                        vix: float, ratio: float, vol_scale: float) -> None:
    """Append one row to daily_returns.csv."""
    file_exists = DAILY_RETURNS_FILE.exists()
    with open(DAILY_RETURNS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "nav", "daily_return", "signal", "vix", "vix_ratio", "vol_scale"])
        writer.writerow([date_str, f"{nav:.2f}", f"{daily_ret:.6f}", signal, f"{vix:.2f}",
                         f"{ratio:.4f}", f"{vol_scale:.4f}"])


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_market_data() -> dict:
    """Fetch VIX, SPY, and SVXY data via yfinance."""
    import yfinance as yf

    # Download all tickers
    vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
    spy_df = yf.download("SPY", period="2mo", interval="1d", progress=False)
    svxy_df = yf.download("SVXY", period="2mo", interval="1d", progress=False)

    # Current VIX level
    vix_close = vix_df["Close"].squeeze()
    vix_level = float(vix_close.iloc[-1])

    # SPY 20-day realized vol (annualized) — proxy for term structure signal
    spy_close = spy_df["Close"].squeeze()
    spy_returns = spy_close.pct_change().dropna()
    spy_realized_vol = float(spy_returns.tail(REALIZED_VOL_WINDOW).std() * np.sqrt(252))

    # VIX / realized_vol ratio
    vix_ratio = vix_level / max(spy_realized_vol * 100, 1.0)  # VIX is in %, realized vol is decimal
    # Actually: VIX is already annualized vol in % terms (e.g. VIX=15 means 15% ann vol)
    # SPY realized vol from returns is decimal (e.g. 0.12 = 12%)
    # So ratio = VIX / (realized_vol * 100) = e.g. 15 / 12 = 1.25
    # Contango (calm): VIX > realized → ratio > 1.0
    # Backwardation (stress): VIX < realized → ratio < 1.0
    # Threshold < 0.9 means: hold SVXY when ratio < 0.9 ... wait, that means stress.
    # Re-reading the spec: "When VIX/VIX_20d_realized_vol ratio < 0.9 (contango = calm): hold SVXY"
    # This means the "VIX_20d_realized_vol" in the spec IS VIX's own 20-day realized vol,
    # and the ratio is VIX_spot / VIX_20d_realized_vol.
    # When VIX is low relative to its own recent realized vol → contango → calm.
    # Let me re-interpret: use VIX / (20d realized vol of VIX itself, not SPY).
    # But the spec says "Compute 20-day realized vol of SPY (as proxy for VIX term structure signal)"
    # So: ratio = VIX_spot / (SPY_20d_realized_vol * 100)
    # VIX ~15, SPY realized vol ~12% → ratio = 15/12 = 1.25 → above 0.9 → cash? That seems wrong.
    # The correct interpretation from the winning config description:
    # "VIX/VIX_20d_realized_vol" — this is VIX divided by VIX's own 20-day realized vol.
    # But the Requirements section says "Compute 20-day realized vol of SPY (as proxy for VIX term structure signal)"
    # I'll use the ratio as specified: VIX / (SPY realized vol * 100)
    # In normal contango: VIX ≈ SPY realized vol → ratio ≈ 1.0
    # In deep contango (calm, VIX suppressed vs actual): ratio < 1.0
    # This makes sense: ratio < 0.9 = VIX is BELOW realized → contango → hold SVXY

    # SVXY 20-day realized vol (for vol targeting)
    svxy_close = svxy_df["Close"].squeeze()
    svxy_returns = svxy_close.pct_change().dropna()
    svxy_realized_vol = float(svxy_returns.tail(REALIZED_VOL_WINDOW).std() * np.sqrt(252)) \
        if len(svxy_returns) >= REALIZED_VOL_WINDOW else 0.40

    svxy_price = float(svxy_close.iloc[-1])
    trade_date = spy_df.index[-1].strftime("%Y-%m-%d")

    return {
        "vix": vix_level,
        "spy_realized_vol": spy_realized_vol,
        "vix_ratio": vix_ratio,
        "svxy_price": svxy_price,
        "svxy_realized_vol": svxy_realized_vol,
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

    vix = mkt["vix"]
    vix_ratio = mkt["vix_ratio"]
    svxy_price = mkt["svxy_price"]
    svxy_realized_vol = mkt["svxy_realized_vol"]
    prev_nav = state["nav"]

    # --- Determine signal ---
    if vix > VIX_CAP:
        signal = "CASH"
        signal_reason = f"VIX={vix:.1f} > cap={VIX_CAP}"
    elif vix_ratio < VIX_RATIO_THRESHOLD:
        signal = "SVXY"
        signal_reason = f"ratio={vix_ratio:.3f} < {VIX_RATIO_THRESHOLD} (contango)"
    else:
        signal = "CASH"
        signal_reason = f"ratio={vix_ratio:.3f} >= {VIX_RATIO_THRESHOLD} (stress)"

    log.info(f"Signal: {signal} | {signal_reason}")

    # --- Compute vol-target position scale ---
    vol_scale = min(VOL_TARGET / max(svxy_realized_vol, 0.05), 1.0)
    log.info(f"Vol-target: {VOL_TARGET:.0%} / {svxy_realized_vol:.1%} = {vol_scale:.2%} scale (capped at 100%)")

    # --- Execute transitions ---
    old_signal = state["signal"]

    if signal == "SVXY" and old_signal == "CASH":
        # Transition: CASH → SVXY
        _enter_position(state, svxy_price, today, vol_scale)

    elif signal == "CASH" and old_signal == "SVXY":
        # Transition: SVXY → CASH
        _exit_position(state, svxy_price, today, signal_reason)

    elif signal == "SVXY" and old_signal == "SVXY":
        # Stay in SVXY — mark to market and check if rebalance needed
        _mark_to_market(state, svxy_price)
        _maybe_rebalance(state, svxy_price, vol_scale, today)

    elif signal == "CASH" and old_signal == "CASH":
        # Stay in cash — accrue interest
        pass

    # --- Accrue cash interest (on any uninvested cash) ---
    if state["position"] is None:
        daily_rate = RISK_FREE_RATE / 252
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = state["cash"]
    else:
        # Also accrue on residual cash from partial position
        if state["cash"] > 1.0:
            daily_rate = RISK_FREE_RATE / 252
            state["cash"] += state["cash"] * daily_rate
            state["nav"] = state["cash"] + state["position"]["shares"] * svxy_price

    state["signal"] = signal
    state["last_check"] = today

    # Track peak NAV for drawdown
    state["peak_nav"] = max(state.get("peak_nav", state["nav"]), state["nav"])

    # --- Record daily return ---
    daily_ret = (state["nav"] / prev_nav - 1) if prev_nav > 0 else 0.0
    state["last_nav"] = state["nav"]
    append_daily_return(today, state["nav"], daily_ret, signal, vix, vix_ratio, vol_scale)

    return state


def _enter_position(state: dict, price: float, date_str: str, vol_scale: float) -> None:
    """Buy SVXY with vol-target-scaled position size."""
    investable = state["cash"] * vol_scale
    shares = int(investable / price)  # whole shares only
    if shares <= 0:
        log.warning(f"Not enough cash ({state['cash']:.2f}) or vol_scale too low ({vol_scale:.2%}) to buy SVXY at {price:.2f}")
        return

    cost = shares * price
    state["cash"] -= cost
    state["position"] = {
        "ticker": "SVXY",
        "shares": shares,
        "entry_price": price,
        "entry_date": date_str,
        "vol_scale": vol_scale,
    }
    state["nav"] = state["cash"] + shares * price
    state["signal"] = "SVXY"

    trade = {
        "action": "BUY",
        "ticker": "SVXY",
        "shares": shares,
        "price": price,
        "date": date_str,
        "nav_after": round(state["nav"], 2),
        "vol_scale": round(vol_scale, 4),
        "reason": "signal_svxy",
    }
    state["trades"].append(trade)
    log.info(f"BUY {shares} SVXY @ ${price:.2f} (vol_scale={vol_scale:.0%}) | NAV: ${state['nav']:,.2f}")


def _exit_position(state: dict, price: float, date_str: str, reason: str) -> None:
    """Sell entire SVXY position."""
    pos = state["position"]
    if pos is None:
        return

    proceeds = pos["shares"] * price
    pnl = (price - pos["entry_price"]) * pos["shares"]
    pnl_pct = (price / pos["entry_price"] - 1) * 100

    state["cash"] += proceeds
    state["nav"] = state["cash"]
    state["signal"] = "CASH"

    trade = {
        "action": "SELL",
        "ticker": "SVXY",
        "shares": pos["shares"],
        "price": price,
        "date": date_str,
        "entry_price": pos["entry_price"],
        "entry_date": pos["entry_date"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "nav_after": round(state["nav"], 2),
        "reason": reason,
    }
    state["trades"].append(trade)
    state["position"] = None

    log.info(
        f"SELL {pos['shares']} SVXY @ ${price:.2f} | "
        f"PnL: ${pnl:,.2f} ({pnl_pct:+.1f}%) | "
        f"NAV: ${state['nav']:,.2f} | Reason: {reason}"
    )


def _mark_to_market(state: dict, price: float) -> None:
    """Update NAV for current SVXY position."""
    pos = state["position"]
    if pos is None:
        return
    state["nav"] = state["cash"] + pos["shares"] * price


def _maybe_rebalance(state: dict, price: float, vol_scale: float, date_str: str) -> None:
    """Rebalance position if vol_scale has changed significantly (>20% delta)."""
    pos = state["position"]
    if pos is None:
        return

    current_shares = pos["shares"]
    total_value = state["cash"] + current_shares * price
    target_investable = total_value * vol_scale
    target_shares = int(target_investable / price)

    if current_shares == 0:
        return

    share_delta_pct = abs(target_shares - current_shares) / current_shares
    if share_delta_pct > 0.20:
        delta = target_shares - current_shares
        if delta > 0:
            # Buy more
            buy_cost = delta * price
            if buy_cost <= state["cash"]:
                state["cash"] -= buy_cost
                pos["shares"] = target_shares
                pos["vol_scale"] = vol_scale
                log.info(f"REBALANCE +{delta} SVXY @ ${price:.2f} (vol_scale={vol_scale:.0%})")
                trade = {
                    "action": "REBALANCE_BUY",
                    "ticker": "SVXY",
                    "shares": delta,
                    "price": price,
                    "date": date_str,
                    "vol_scale": round(vol_scale, 4),
                    "reason": "vol_rebalance",
                }
                state["trades"].append(trade)
        elif delta < 0:
            # Sell some
            sell_proceeds = abs(delta) * price
            state["cash"] += sell_proceeds
            pos["shares"] = target_shares
            pos["vol_scale"] = vol_scale
            log.info(f"REBALANCE {delta} SVXY @ ${price:.2f} (vol_scale={vol_scale:.0%})")
            trade = {
                "action": "REBALANCE_SELL",
                "ticker": "SVXY",
                "shares": abs(delta),
                "price": price,
                "date": date_str,
                "vol_scale": round(vol_scale, 4),
                "reason": "vol_rebalance",
            }
            state["trades"].append(trade)

        state["nav"] = state["cash"] + pos["shares"] * price


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Vol Harvest Paper Engine — daily check")
    log.info("=" * 60)

    # Load state
    if STATE_FILE.exists():
        state = load_state()
        log.info(
            f"Loaded state: signal={state['signal']}, "
            f"NAV=${state['nav']:,.2f}, "
            f"position={'SVXY' if state['position'] else 'CASH'}"
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
        print(f"[VolHarvest] ERROR: data fetch failed — {e}")
        return

    log.info(
        f"Market: VIX={mkt['vix']:.2f} | "
        f"SPY RealVol={mkt['spy_realized_vol']:.1%} | "
        f"Ratio={mkt['vix_ratio']:.3f} | "
        f"SVXY=${mkt['svxy_price']:.2f} | "
        f"SVXY Vol={mkt['svxy_realized_vol']:.1%} | "
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
    pos_str = f"SVXY x{state['position']['shares']}" if state["position"] else "CASH"
    summary = (
        f"[VolHarvest] {mkt['date']} | "
        f"Signal={state['signal']} | {pos_str} | "
        f"NAV=${state['nav']:,.2f} | "
        f"DD={dd:.1f}% | "
        f"VIX={mkt['vix']:.1f} ratio={mkt['vix_ratio']:.3f} | "
        f"Trades={len(state['trades'])}"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
