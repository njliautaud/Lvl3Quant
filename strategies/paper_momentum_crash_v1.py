"""
Momentum + Crash Filter Paper Engine — Sector Rotation with Crash Protection.

Strategy (winner from walk-forward growth research):
  - Monthly rebalance, long-only.
  - Universe: 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLU, XLRE, XLB).
  - Signal: rank sectors by trailing 6-month momentum (total return). Hold top 3.
  - Equal-weight the top 3 sectors (1/3 each).
  - Crash filter (any ONE triggers full exit to cash or 50% reduction):
      * VIX > 28
      * Credit spread proxy: HYG/IEF ratio drops > 2% over trailing 20 days
      * SPY below its 200-day SMA
  - Rebalance on the first trading day of each month.
  - On non-rebalance days: only check crash filter for emergency exit.
  - Re-enter on next month-start rebalance if crash filter clears.
  - Commission-free (Robinhood, HC #694).
  - Cash earns 5% annualized risk-free rate.

Runs daily at 16:15 ET via PM2 cron. Idempotent (safe to run twice same day).

Outputs:
  - data/paper_engines/momentum_crash/state.json
  - data/paper_engines/momentum_crash/trades.json
  - data/paper_engines/momentum_crash/daily_returns.csv
  - logs/momentum_crash_paper.log
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
STATE_DIR = ROOT / "data" / "paper_engines" / "momentum_crash"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.json"
DAILY_RETURNS_FILE = STATE_DIR / "daily_returns.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
TOP_N = 3                       # hold top 3 sectors
MOMENTUM_LOOKBACK_DAYS = 126    # ~6 months of trading days
SMA_PERIOD = 200                # SPY 200-day SMA for crash filter
VIX_CRASH_THRESHOLD = 28.0      # VIX > this triggers crash filter
CREDIT_SPREAD_DROP_PCT = 0.02   # HYG/IEF ratio drop > 2% in 20d triggers crash
CREDIT_SPREAD_WINDOW = 20       # trailing days for credit spread check
INITIAL_NAV = 100_000.0
RISK_FREE_RATE = 0.05           # 5% annual for cash
CRASH_ACTION = "full_exit"      # "full_exit" or "half_exit" (reduce to 50%)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MomCrash] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "momentum_crash_paper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("momentum_crash")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def default_state() -> dict:
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "positions": [],          # list of {"ticker", "shares", "entry_price", "entry_date"}
        "crash_filter_active": False,
        "crash_reasons": [],
        "last_check": None,
        "last_rebalance_month": None,  # "YYYY-MM" of last rebalance
        "last_nav": INITIAL_NAV,
        "inception_date": today_str,
        "peak_nav": INITIAL_NAV,
        "momentum_rankings": [],   # latest rankings for logging
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
                        signal: str, crash_active: bool) -> None:
    """Append one row to daily_returns.csv."""
    file_exists = DAILY_RETURNS_FILE.exists()
    with open(DAILY_RETURNS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "nav", "daily_return", "holdings", "crash_filter"])
        writer.writerow([date_str, f"{nav:.2f}", f"{daily_ret:.6f}", signal,
                         "ACTIVE" if crash_active else "OFF"])


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_market_data() -> dict:
    """Fetch all required data via yfinance."""
    import yfinance as yf

    # Download sector ETFs — need 7+ months for 6-month momentum
    tickers_str = " ".join(SECTOR_ETFS)
    sector_data = yf.download(tickers_str, period="8mo", interval="1d", progress=False)

    # SPY for SMA200 crash filter (need 1+ year)
    spy_df = yf.download("SPY", period="2y", interval="1d", progress=False)
    spy_close = spy_df["Close"].squeeze()
    spy_sma200 = spy_close.rolling(SMA_PERIOD).mean()

    # VIX for crash filter
    vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
    vix_close = vix_df["Close"].squeeze()

    # HYG and IEF for credit spread proxy
    credit_df = yf.download("HYG IEF", period="2mo", interval="1d", progress=False)

    trade_date = spy_df.index[-1].strftime("%Y-%m-%d")

    # Compute momentum rankings
    sector_close = sector_data["Close"]
    momentum_rankings = []
    for etf in SECTOR_ETFS:
        try:
            prices = sector_close[etf].dropna()
            if len(prices) < MOMENTUM_LOOKBACK_DAYS:
                log.warning(f"{etf}: only {len(prices)} days of data, need {MOMENTUM_LOOKBACK_DAYS}")
                momentum_rankings.append((etf, -999.0))
                continue
            current = float(prices.iloc[-1])
            lookback = float(prices.iloc[-MOMENTUM_LOOKBACK_DAYS])
            mom = (current / lookback - 1) * 100  # percent return
            momentum_rankings.append((etf, mom))
        except Exception as e:
            log.warning(f"{etf}: momentum calc failed — {e}")
            momentum_rankings.append((etf, -999.0))

    # Sort by momentum descending
    momentum_rankings.sort(key=lambda x: x[1], reverse=True)

    # Current sector prices (for position sizing)
    sector_prices = {}
    for etf in SECTOR_ETFS:
        try:
            sector_prices[etf] = float(sector_close[etf].dropna().iloc[-1])
        except Exception:
            sector_prices[etf] = None

    # Crash filter components
    spy_price = float(spy_close.iloc[-1])
    sma200_val = float(spy_sma200.iloc[-1])
    vix_level = float(vix_close.iloc[-1])

    # Credit spread: HYG/IEF ratio, check 20d change
    hyg_close = credit_df["Close"]["HYG"].dropna()
    ief_close = credit_df["Close"]["IEF"].dropna()
    # Align dates
    common_idx = hyg_close.index.intersection(ief_close.index)
    hyg_aligned = hyg_close.loc[common_idx]
    ief_aligned = ief_close.loc[common_idx]
    credit_ratio = hyg_aligned / ief_aligned

    credit_ratio_current = float(credit_ratio.iloc[-1])
    if len(credit_ratio) >= CREDIT_SPREAD_WINDOW:
        credit_ratio_20d_ago = float(credit_ratio.iloc[-CREDIT_SPREAD_WINDOW])
        credit_ratio_change = (credit_ratio_current / credit_ratio_20d_ago - 1)
    else:
        credit_ratio_change = 0.0

    return {
        "date": trade_date,
        "momentum_rankings": momentum_rankings,
        "sector_prices": sector_prices,
        "spy_close": spy_price,
        "spy_sma200": sma200_val,
        "spy_above_sma": spy_price > sma200_val,
        "vix": vix_level,
        "credit_ratio": credit_ratio_current,
        "credit_ratio_change": credit_ratio_change,
    }


# ---------------------------------------------------------------------------
# Crash filter
# ---------------------------------------------------------------------------
def check_crash_filter(mkt: dict) -> tuple[bool, list[str]]:
    """Check all crash filter conditions. Returns (is_crash, list_of_reasons)."""
    reasons = []

    if mkt["vix"] > VIX_CRASH_THRESHOLD:
        reasons.append(f"VIX={mkt['vix']:.1f} > {VIX_CRASH_THRESHOLD}")

    if mkt["credit_ratio_change"] < -CREDIT_SPREAD_DROP_PCT:
        reasons.append(
            f"HYG/IEF ratio dropped {mkt['credit_ratio_change']*100:.1f}% "
            f"in {CREDIT_SPREAD_WINDOW}d (threshold: -{CREDIT_SPREAD_DROP_PCT*100:.0f}%)"
        )

    if not mkt["spy_above_sma"]:
        reasons.append(
            f"SPY={mkt['spy_close']:.2f} below 200d SMA={mkt['spy_sma200']:.2f}"
        )

    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Rebalance logic
# ---------------------------------------------------------------------------
def is_month_start_rebalance(trade_date: str, last_rebalance_month: str | None) -> bool:
    """Check if today is the first trading day of a new month we haven't rebalanced for."""
    current_month = trade_date[:7]  # "YYYY-MM"
    return current_month != last_rebalance_month


def compute_nav(state: dict, sector_prices: dict) -> float:
    """Compute current NAV from positions + cash."""
    nav = state["cash"]
    for pos in state["positions"]:
        price = sector_prices.get(pos["ticker"])
        if price:
            nav += pos["shares"] * price
    return nav


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def process_day(state: dict, mkt: dict) -> dict:
    """Run one daily check. Modifies state in place."""
    today = mkt["date"]

    # Idempotency
    if state["last_check"] == today:
        log.info(f"Already processed {today}, skipping.")
        return state

    prev_nav = state["nav"]

    # Mark to market first
    state["nav"] = compute_nav(state, mkt["sector_prices"])

    # Check crash filter
    crash_active, crash_reasons = check_crash_filter(mkt)
    was_crash = state["crash_filter_active"]

    if crash_active:
        log.info(f"CRASH FILTER ACTIVE: {'; '.join(crash_reasons)}")
        state["crash_filter_active"] = True
        state["crash_reasons"] = crash_reasons

        # If we have positions, exit
        if state["positions"]:
            if CRASH_ACTION == "full_exit":
                _exit_all_positions(state, mkt["sector_prices"], today, "crash_filter")
            else:
                _reduce_positions(state, mkt["sector_prices"], today, 0.50, "crash_filter_half")
    else:
        state["crash_filter_active"] = False
        state["crash_reasons"] = []

        # Check if this is a month-start rebalance day
        if is_month_start_rebalance(today, state["last_rebalance_month"]):
            log.info(f"MONTH-START REBALANCE — {today[:7]}")
            _rebalance(state, mkt, today)
            state["last_rebalance_month"] = today[:7]
        else:
            # Non-rebalance day, crash filter clear — just hold
            if state["positions"]:
                holdings = ", ".join(p["ticker"] for p in state["positions"])
                log.info(f"Holding: {holdings} | NAV: ${state['nav']:,.2f}")
            else:
                log.info(f"In cash | NAV: ${state['nav']:,.2f}")

    # Accrue cash interest
    if state["cash"] > 1.0:
        daily_rate = RISK_FREE_RATE / 252
        interest = state["cash"] * daily_rate
        state["cash"] += interest
        state["nav"] = compute_nav(state, mkt["sector_prices"])

    # Track peak NAV
    state["peak_nav"] = max(state.get("peak_nav", state["nav"]), state["nav"])
    state["momentum_rankings"] = mkt["momentum_rankings"]
    state["last_check"] = today

    # Record daily return
    daily_ret = (state["nav"] / prev_nav - 1) if prev_nav > 0 else 0.0
    state["last_nav"] = state["nav"]
    holdings_str = "+".join(p["ticker"] for p in state["positions"]) if state["positions"] else "CASH"
    append_daily_return(today, state["nav"], daily_ret, holdings_str, crash_active)

    return state


def _rebalance(state: dict, mkt: dict, date_str: str) -> None:
    """Sell all current positions and buy top-N sectors by momentum."""
    # Exit everything first
    if state["positions"]:
        _exit_all_positions(state, mkt["sector_prices"], date_str, "monthly_rebalance")

    # Select top N sectors
    top_sectors = []
    for etf, mom in mkt["momentum_rankings"][:TOP_N]:
        price = mkt["sector_prices"].get(etf)
        if price and mom > -999:
            top_sectors.append((etf, price, mom))

    if not top_sectors:
        log.warning("No valid sectors to buy — staying in cash.")
        return

    log.info(f"Top {TOP_N} sectors: {[(s, f'{m:.1f}%') for s, _, m in top_sectors]}")

    # Equal-weight allocation
    alloc_per_sector = state["cash"] / len(top_sectors)
    for etf, price, mom in top_sectors:
        shares = int(alloc_per_sector / price)
        if shares <= 0:
            log.warning(f"Cannot afford {etf} at ${price:.2f} with ${alloc_per_sector:.2f}")
            continue

        cost = shares * price
        state["cash"] -= cost
        pos = {
            "ticker": etf,
            "shares": shares,
            "entry_price": price,
            "entry_date": date_str,
            "momentum_pct": round(mom, 2),
        }
        state["positions"].append(pos)

        trade = {
            "action": "BUY",
            "ticker": etf,
            "shares": shares,
            "price": price,
            "date": date_str,
            "nav_after": round(compute_nav(state, mkt["sector_prices"]), 2),
            "reason": f"momentum_rank_top{TOP_N}",
            "momentum_pct": round(mom, 2),
        }
        state["trades"].append(trade)
        log.info(f"BUY {shares} {etf} @ ${price:.2f} (6mo mom: {mom:.1f}%)")

    state["nav"] = compute_nav(state, mkt["sector_prices"])
    log.info(f"Rebalanced — NAV: ${state['nav']:,.2f}")


def _exit_all_positions(state: dict, sector_prices: dict, date_str: str, reason: str) -> None:
    """Sell all positions."""
    for pos in state["positions"]:
        price = sector_prices.get(pos["ticker"])
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
            "nav_after": round(state["cash"], 2),
            "reason": reason,
        }
        state["trades"].append(trade)
        log.info(
            f"SELL {pos['shares']} {pos['ticker']} @ ${price:.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Reason: {reason}"
        )

    state["positions"] = []
    state["nav"] = state["cash"]


def _reduce_positions(state: dict, sector_prices: dict, date_str: str,
                      target_pct: float, reason: str) -> None:
    """Reduce all positions to target_pct of current size."""
    new_positions = []
    for pos in state["positions"]:
        price = sector_prices.get(pos["ticker"], pos["entry_price"])
        target_shares = int(pos["shares"] * target_pct)
        sell_shares = pos["shares"] - target_shares

        if sell_shares > 0:
            proceeds = sell_shares * price
            pnl = (price - pos["entry_price"]) * sell_shares
            state["cash"] += proceeds

            trade = {
                "action": "REDUCE",
                "ticker": pos["ticker"],
                "shares": sell_shares,
                "price": price,
                "date": date_str,
                "pnl": round(pnl, 2),
                "reason": reason,
            }
            state["trades"].append(trade)
            log.info(f"REDUCE {sell_shares} {pos['ticker']} @ ${price:.2f} | Reason: {reason}")

        if target_shares > 0:
            new_pos = pos.copy()
            new_pos["shares"] = target_shares
            new_positions.append(new_pos)

    state["positions"] = new_positions
    state["nav"] = compute_nav(state, sector_prices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Momentum + Crash Filter Paper Engine — daily check")
    log.info("=" * 60)

    # Load state
    if STATE_FILE.exists():
        state = load_state()
        holdings = "+".join(p["ticker"] for p in state["positions"]) if state["positions"] else "CASH"
        log.info(
            f"Loaded state: holdings={holdings}, "
            f"NAV=${state['nav']:,.2f}, "
            f"crash_filter={'ACTIVE' if state['crash_filter_active'] else 'OFF'}"
        )
    else:
        log.info("No state file found — creating initial state.")
        state = default_state()
        save_state(state)
        save_trades([])
        log.info(f"Initial state saved: NAV=${state['nav']:,.2f} (CASH)")
        # Fall through to process today

    # Fetch market data
    try:
        mkt = get_market_data()
    except Exception as e:
        log.error(f"Failed to fetch market data: {e}")
        log.warning("Staying in current positions due to data failure.")
        print(f"[MomCrash] ERROR: data fetch failed — {e}")
        return

    # Log market summary
    top3 = mkt["momentum_rankings"][:3]
    top3_str = ", ".join(f"{t}={m:.1f}%" for t, m in top3)
    log.info(
        f"Market: SPY={mkt['spy_close']:.2f} SMA200={mkt['spy_sma200']:.2f} "
        f"{'ABOVE' if mkt['spy_above_sma'] else 'BELOW'} | "
        f"VIX={mkt['vix']:.1f} | "
        f"HYG/IEF Δ20d={mkt['credit_ratio_change']*100:.1f}% | "
        f"Top3: {top3_str} | Date={mkt['date']}"
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
    holdings = "+".join(p["ticker"] for p in state["positions"]) if state["positions"] else "CASH"
    crash_str = "CRASH!" if state["crash_filter_active"] else "OK"
    summary = (
        f"[MomCrash] {mkt['date']} | "
        f"Hold={holdings} | "
        f"NAV=${state['nav']:,.2f} | "
        f"DD={dd:.1f}% | "
        f"Filter={crash_str} | "
        f"VIX={mkt['vix']:.1f} | "
        f"Trades={len(state['trades'])}"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
