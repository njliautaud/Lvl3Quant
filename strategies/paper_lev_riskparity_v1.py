"""
3x Leveraged Risk Parity + Momentum Tilt Paper Engine.

Strategy (from R5 walk-forward research):
  - Assets: SPY, TLT, GLD, DBC (equities, bonds, gold, commodities)
  - Leverage: 3x via leveraged ETFs:
      * UPRO (3x SPY)
      * TMF (3x TLT)
      * UGL (2x GLD — no liquid 3x gold ETF)
      * DBC (1x — no good leveraged commodity ETF)
  - Weights: risk parity base (inverse realized vol, 63d lookback)
    with momentum tilt (12-1 month momentum, overweight positive, underweight negative)
  - Rebalance: monthly on first trading day
  - Starting NAV: $100,000
  - Commission-free (Robinhood, HC #694)
  - Cash earns 5% annualized risk-free rate

Walk-forward research results (R5):
  3x base RP: CAGR 17.9%, Sharpe 0.761, MaxDD -49.1%, Regime gap 0.175 (PASS)
  With momentum tilt: Sharpe ~0.87, improved CAGR, lower MaxDD

Runs daily at 16:15 ET via PM2 cron. Idempotent (safe to run twice same day).

Outputs:
  - data/paper_engines/lev_riskparity/state.json
  - data/paper_engines/lev_riskparity/trades.json
  - data/paper_engines/lev_riskparity/daily_returns.csv
  - logs/lev_riskparity_paper.log
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # /home/jupiter/Lvl3Quant
STATE_DIR = ROOT / "data" / "paper_engines" / "lev_riskparity"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"
DAILY_RETURNS_FILE = STATE_DIR / "daily_returns.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# The underlying assets we track for signals
BASE_ASSETS = ["SPY", "TLT", "GLD", "DBC"]

# What we actually trade (leveraged ETFs)
TRADE_MAP = {
    "SPY": {"etf": "UPRO", "leverage": 3.0},
    "TLT": {"etf": "TMF",  "leverage": 3.0},
    "GLD": {"etf": "UGL",  "leverage": 2.0},
    "DBC": {"etf": "DBC",  "leverage": 1.0},
}

# Risk parity parameters
VOL_LOOKBACK_DAYS = 63           # ~3 months for vol estimation
MOMENTUM_LOOKBACK_FAST = 21      # 1 month (skip in 12-1)
MOMENTUM_LOOKBACK_SLOW = 252     # 12 months

INITIAL_NAV = 100_000.0
RISK_FREE_RATE = 0.05            # 5% annual for cash

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LevRP] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "lev_riskparity_paper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lev_riskparity")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "positions": [],          # list of {"ticker", "shares", "entry_price", "entry_date", "base_asset", "weight"}
        "last_check": None,
        "last_rebalance_month": None,  # "YYYY-MM" of last rebalance
        "last_nav": INITIAL_NAV,
        "inception_date": today_str,
        "peak_nav": INITIAL_NAV,
        "weights_detail": {},      # latest weight computation detail
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


def append_daily_return(date_str: str, nav: float, daily_ret: float,
                        holdings_str: str) -> None:
    """Append one row to daily_returns.csv."""
    file_exists = DAILY_RETURNS_FILE.exists()
    with open(DAILY_RETURNS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "nav", "daily_return", "holdings"])
        writer.writerow([date_str, f"{nav:.2f}", f"{daily_ret:.6f}", holdings_str])


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_market_data() -> dict:
    """Fetch all required data via yfinance."""
    import yfinance as yf

    # Download base assets — need 13+ months for 12-1 momentum
    base_str = " ".join(BASE_ASSETS)
    base_data = yf.download(base_str, period="15mo", interval="1d", progress=False)

    # Download leveraged ETFs for current prices
    lev_tickers = [TRADE_MAP[a]["etf"] for a in BASE_ASSETS]
    lev_str = " ".join(lev_tickers)
    lev_data = yf.download(lev_str, period="5d", interval="1d", progress=False)

    trade_date = base_data.index[-1].strftime("%Y-%m-%d")

    # Base asset close prices (for signal computation)
    base_close = base_data["Close"]

    # Compute realized vol (63d) for each base asset
    base_returns = base_close.pct_change()
    realized_vols = {}
    for asset in BASE_ASSETS:
        try:
            rets = base_returns[asset].dropna()
            if len(rets) >= VOL_LOOKBACK_DAYS:
                vol = float(rets.iloc[-VOL_LOOKBACK_DAYS:].std() * np.sqrt(252))
                realized_vols[asset] = vol
            else:
                log.warning(f"{asset}: only {len(rets)} days, need {VOL_LOOKBACK_DAYS}")
                realized_vols[asset] = 0.20  # default 20% vol
        except Exception as e:
            log.warning(f"{asset}: vol calc failed — {e}")
            realized_vols[asset] = 0.20

    # Compute 12-1 month momentum for each base asset
    momentums = {}
    for asset in BASE_ASSETS:
        try:
            prices = base_close[asset].dropna()
            if len(prices) >= MOMENTUM_LOOKBACK_SLOW:
                # 12-month return minus 1-month return (skip recent month)
                price_now = float(prices.iloc[-MOMENTUM_LOOKBACK_FAST])  # 1 month ago
                price_12m = float(prices.iloc[-MOMENTUM_LOOKBACK_SLOW])  # 12 months ago
                mom = (price_now / price_12m - 1) * 100
                momentums[asset] = mom
            elif len(prices) >= MOMENTUM_LOOKBACK_FAST * 6:
                # Fallback: use whatever we have
                half = len(prices) // 2
                price_now = float(prices.iloc[-MOMENTUM_LOOKBACK_FAST])
                price_past = float(prices.iloc[0])
                mom = (price_now / price_past - 1) * 100
                momentums[asset] = mom
            else:
                momentums[asset] = 0.0
        except Exception as e:
            log.warning(f"{asset}: momentum calc failed — {e}")
            momentums[asset] = 0.0

    # Current leveraged ETF prices
    lev_prices = {}
    lev_close = lev_data["Close"]
    for asset in BASE_ASSETS:
        etf = TRADE_MAP[asset]["etf"]
        try:
            lev_prices[etf] = float(lev_close[etf].dropna().iloc[-1])
        except Exception:
            lev_prices[etf] = None

    return {
        "date": trade_date,
        "realized_vols": realized_vols,
        "momentums": momentums,
        "lev_prices": lev_prices,
    }


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------
def compute_risk_parity_momentum_weights(realized_vols: dict, momentums: dict) -> dict:
    """
    Compute target weights using inverse-vol risk parity with momentum tilt.

    1. Base: inverse vol weights (risk parity)
    2. Tilt: overweight assets with positive 12-1 momentum, underweight negative
    3. Weights sum to 1.0 (leverage is embedded in the ETFs)
    """
    # Step 1: Inverse vol weights
    inv_vols = {}
    for asset in BASE_ASSETS:
        vol = realized_vols.get(asset, 0.20)
        if vol <= 0:
            vol = 0.20
        inv_vols[asset] = 1.0 / vol

    total_inv_vol = sum(inv_vols.values())
    base_weights = {a: iv / total_inv_vol for a, iv in inv_vols.items()}

    # Step 2: Momentum tilt
    # Rank assets by momentum, apply multiplicative tilt
    mom_values = [(a, momentums.get(a, 0.0)) for a in BASE_ASSETS]
    mom_values.sort(key=lambda x: x[1])

    # Assign ranks 1-N
    n = len(mom_values)
    ranks = {}
    for i, (asset, _) in enumerate(mom_values):
        ranks[asset] = i + 1  # 1 = worst momentum, N = best

    # Tilt factor: rank-based, centered at 1.0
    # Range: [0.7, 1.3] for 4 assets
    tilt_factors = {}
    mean_rank = (n + 1) / 2
    for asset in BASE_ASSETS:
        rank = ranks[asset]
        # Linear tilt: +-30% based on rank
        tilt = 1.0 + (rank - mean_rank) / (n - 1) * 0.6
        tilt_factors[asset] = np.clip(tilt, 0.5, 1.5)

    # Apply tilt
    tilted_weights = {a: base_weights[a] * tilt_factors[a] for a in BASE_ASSETS}

    # Re-normalize to sum to 1.0
    total = sum(tilted_weights.values())
    final_weights = {a: w / total for a, w in tilted_weights.items()}

    detail = {
        "base_weights": {a: round(w, 4) for a, w in base_weights.items()},
        "momentum_pct": {a: round(momentums.get(a, 0), 2) for a in BASE_ASSETS},
        "momentum_ranks": ranks,
        "tilt_factors": {a: round(tf, 3) for a, tf in tilt_factors.items()},
        "final_weights": {a: round(w, 4) for a, w in final_weights.items()},
        "realized_vols": {a: round(v, 4) for a, v in realized_vols.items()},
    }

    return final_weights, detail


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def compute_nav(state: dict, lev_prices: dict) -> float:
    """Compute current NAV from positions + cash."""
    nav = state["cash"]
    for pos in state["positions"]:
        price = lev_prices.get(pos["ticker"])
        if price:
            nav += pos["shares"] * price
    return nav


def is_month_start_rebalance(trade_date: str, last_rebalance_month: str | None) -> bool:
    """Check if today is the first trading day of a new month we haven't rebalanced for."""
    current_month = trade_date[:7]  # "YYYY-MM"
    return current_month != last_rebalance_month


def process_day(state: dict, mkt: dict) -> dict:
    """Run one daily check. Modifies state in place."""
    today = mkt["date"]

    # Idempotency
    if state["last_check"] == today:
        log.info(f"Already processed {today}, skipping.")
        return state

    prev_nav = state["nav"]

    # Mark to market
    state["nav"] = compute_nav(state, mkt["lev_prices"])

    # Check if this is a month-start rebalance day
    if is_month_start_rebalance(today, state["last_rebalance_month"]):
        log.info(f"MONTH-START REBALANCE — {today[:7]}")
        _rebalance(state, mkt, today)
        state["last_rebalance_month"] = today[:7]
    else:
        # Non-rebalance day — just hold
        if state["positions"]:
            holdings = ", ".join(f"{p['ticker']}({p['shares']})" for p in state["positions"])
            log.info(f"Holding: {holdings} | NAV: ${state['nav']:,.2f}")
        else:
            log.info(f"In cash (pre-first-rebalance) | NAV: ${state['nav']:,.2f}")

    # Accrue cash interest
    if state["cash"] > 1.0:
        daily_rate = RISK_FREE_RATE / 252
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = compute_nav(state, mkt["lev_prices"])

    # Track peak NAV
    state["peak_nav"] = max(state.get("peak_nav", state["nav"]), state["nav"])
    state["last_check"] = today

    # Record daily return
    daily_ret = (state["nav"] / prev_nav - 1) if prev_nav > 0 else 0.0
    state["last_nav"] = state["nav"]
    holdings_str = "+".join(p["ticker"] for p in state["positions"]) if state["positions"] else "CASH"
    append_daily_return(today, state["nav"], daily_ret, holdings_str)

    return state


def _rebalance(state: dict, mkt: dict, date_str: str) -> None:
    """Sell all positions and rebalance to target risk parity + momentum weights."""
    # Exit everything first
    if state["positions"]:
        _exit_all_positions(state, mkt["lev_prices"], date_str, "monthly_rebalance")

    # Compute target weights
    final_weights, detail = compute_risk_parity_momentum_weights(
        mkt["realized_vols"], mkt["momentums"]
    )
    state["weights_detail"] = detail

    wts_str = ", ".join(f"{a}={w:.1%}" for a, w in final_weights.items())
    vols_str = ", ".join(f"{a}={mkt['realized_vols'].get(a,0):.1%}" for a in BASE_ASSETS)
    mom_str = ", ".join(f"{a}={mkt['momentums'].get(a,0):+.1f}%" for a in BASE_ASSETS)
    log.info(f"Target weights: {wts_str}")
    log.info(f"Vols: {vols_str}")
    log.info(f"Mom: {mom_str}")

    # Allocate to leveraged ETFs
    total_cash = state["cash"]
    for asset in BASE_ASSETS:
        weight = final_weights[asset]
        etf = TRADE_MAP[asset]["etf"]
        lev = TRADE_MAP[asset]["leverage"]
        price = mkt["lev_prices"].get(etf)

        if price is None or price <= 0:
            log.warning(f"No price for {etf}, skipping")
            continue

        # Allocation: weight * total NAV
        alloc = weight * total_cash

        # Buy fractional-share-friendly integer shares
        shares = int(alloc / price)
        if shares <= 0:
            log.warning(f"Cannot afford {etf} at ${price:.2f} with ${alloc:.2f}")
            continue

        cost = shares * price
        state["cash"] -= cost
        pos = {
            "ticker": etf,
            "shares": shares,
            "entry_price": price,
            "entry_date": date_str,
            "base_asset": asset,
            "weight": round(weight, 4),
            "leverage": lev,
        }
        state["positions"].append(pos)

        trade = {
            "action": "BUY",
            "ticker": etf,
            "shares": shares,
            "price": price,
            "date": date_str,
            "base_asset": asset,
            "weight": round(weight, 4),
            "leverage": lev,
            "reason": "monthly_rebalance",
        }
        state["trades"].append(trade)
        log.info(
            f"BUY {shares} {etf} @ ${price:.2f} "
            f"(base={asset}, wt={weight:.1%}, lev={lev}x)"
        )

    state["nav"] = compute_nav(state, mkt["lev_prices"])
    log.info(f"Rebalanced — NAV: ${state['nav']:,.2f}, residual cash: ${state['cash']:,.2f}")


def _exit_all_positions(state: dict, lev_prices: dict, date_str: str, reason: str) -> None:
    """Sell all positions."""
    for pos in state["positions"]:
        price = lev_prices.get(pos["ticker"])
        if price is None:
            log.error(f"No price for {pos['ticker']} — using entry price")
            price = pos["entry_price"]

        proceeds = pos["shares"] * price
        pnl = (price - pos["entry_price"]) * pos["shares"]
        pnl_pct = (price / pos["entry_price"] - 1) * 100

        state["cash"] += proceeds

        trade = {
            "action": "SELL",
            "ticker": pos["ticker"],
            "shares": pos["shares"],
            "price": price,
            "date": date_str,
            "entry_price": pos["entry_price"],
            "entry_date": pos["entry_date"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
        }
        state["trades"].append(trade)
        log.info(
            f"SELL {pos['shares']} {pos['ticker']} @ ${price:.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Reason: {reason}"
        )

    state["positions"] = []
    state["nav"] = state["cash"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("3x Leveraged Risk Parity + Momentum Tilt — daily check")
    log.info("=" * 60)

    # Load state
    if STATE_FILE.exists():
        state = load_state()
        holdings = "+".join(f"{p['ticker']}({p['shares']})" for p in state["positions"]) if state["positions"] else "CASH"
        log.info(
            f"Loaded state: holdings={holdings}, "
            f"NAV=${state['nav']:,.2f}"
        )
    else:
        log.info("No state file found — creating initial state.")
        state = default_state()
        save_state(state)
        save_trades([])
        log.info(f"Initial state saved: NAV=${state['nav']:,.2f} (CASH)")

    # Fetch market data
    try:
        mkt = get_market_data()
    except Exception as e:
        log.error(f"Failed to fetch market data: {e}")
        log.warning("Staying in current positions due to data failure.")
        print(f"[LevRP] ERROR: data fetch failed — {e}")
        return

    # Log market summary
    vols_str = ", ".join(f"{a}={mkt['realized_vols'].get(a,0):.1%}" for a in BASE_ASSETS)
    mom_str = ", ".join(f"{a}={mkt['momentums'].get(a,0):+.1f}%" for a in BASE_ASSETS)
    log.info(f"Vols: {vols_str}")
    log.info(f"Momentum: {mom_str}")
    log.info(f"Date: {mkt['date']}")

    # Log leveraged ETF prices
    for asset in BASE_ASSETS:
        etf = TRADE_MAP[asset]["etf"]
        price = mkt["lev_prices"].get(etf, "N/A")
        log.info(f"  {etf} ({asset} {TRADE_MAP[asset]['leverage']}x): ${price}")

    # Process
    state = process_day(state, mkt)

    # Save
    save_state(state)
    save_trades(state["trades"])

    # Compute drawdown
    peak = state.get("peak_nav", state["nav"])
    dd = (state["nav"] / peak - 1) * 100 if peak > 0 else 0.0

    # One-line PM2 summary
    holdings = "+".join(p["ticker"] for p in state["positions"]) if state["positions"] else "CASH"
    summary = (
        f"[LevRP] {mkt['date']} | "
        f"Hold={holdings} | "
        f"NAV=${state['nav']:,.2f} | "
        f"DD={dd:.1f}% | "
        f"Trades={len(state['trades'])}"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
