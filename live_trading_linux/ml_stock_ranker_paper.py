#!/usr/bin/env python3
"""
ML Asymmetric Stock Ranker — Paper Trading Engine.

Strategy B upgraded with 3m features (validated, permutation test p=0.000):
  - Original 1m-only: Sharpe 0.72, CAGR 14.3%, FAILS regime gate
  - Upgraded 1m+3m: Sharpe 0.97, CAGR 20.6%, PASSES regime gate
  - 3m-only: Sharpe 1.13, CAGR 20.7%, best regime balance (bull 1.10, bear 1.36)
  - Monthly rebalance on 1st trading day of month.
  - Universe: 50 large-cap US stocks.
  - Asymmetric filter: fires when >=3 stocks have high vol + negative momentum
    + volume surge. Fires ~26% of months historically.
  - When filter fires: rank stocks by composite distress score using both
    1m features (vol_63d, vix, mom_1m, dist_52w_high) and 3m features
    (mom_3m, vol_126d, dist_200sma), buy top 5 equal weight.
  - When filter doesn't fire: hold SPY.
  - Hold for 1 month until next rebalance.

Note: uses rules-based composite z-score approximation instead of trained
LightGBM (weights not available on Jupiter). The z-score captures the same
feature intuition, now including 3-month horizon features for regime robustness.

Run mode: cron-driven, one invocation per scheduled run.
  - On rebalance day: compute features, check filter, rebalance portfolio.
  - On non-rebalance day (manual run): mark-to-market NAV only.

State: live_trading_linux/state/ml_stock_ranker_state.json
Log:   logs/ml_stock_ranker_paper.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "live_trading_linux" / "state"
STATE_FILE = STATE_DIR / "ml_stock_ranker_state.json"
LOG_FILE = ROOT / "logs" / "ml_stock_ranker_paper.log"
TRADES_LOG = STATE_DIR / "ml_stock_ranker_trades.jsonl"

STATE_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(parents=True, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ml_stock_ranker")

# ── Config ──────────────────────────────────────────────────────────────────
INITIAL_NAV = 100_000.0
TOP_K = 5  # number of stocks to hold when filter fires
COST_BPS = 10.0  # round-trip cost estimate (commission + slippage)
MIN_FILTER_STOCKS = 3  # asymmetric filter fires if >= this many stocks pass

# Asymmetric filter thresholds
FILTER_VOL_PCTILE = 80  # vol_20d must be > 80th percentile within universe
FILTER_MOM_3M_THRESH = -0.05  # momentum 3m < -5%
FILTER_VOLUME_SURGE = 1.5  # volume > 1.5x 20-day average

# 50-stock large-cap universe
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT",
    "ACN", "CSCO", "ABT", "DHR", "CRM", "NKE", "TXN", "NEE", "UPS",
    "LIN", "AMD", "QCOM", "HON", "LOW", "AMGN", "INTC", "BA", "GS",
    "CAT", "BLK", "ISRG", "SYK", "ADP",
]

BENCH = "SPY"


# ── State Management ───────────────────────────────────────────────────────
def load_state() -> dict:
    """Load or initialize paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "nav": INITIAL_NAV,
        "cash": INITIAL_NAV,
        "positions": {},  # {ticker: {"shares": float, "entry_price": float, "entry_date": str}}
        "mode": "IDLE",  # IDLE | SPY | PICKS
        "last_rebalance_date": None,
        "filter_fired": False,
        "filter_stocks_passing": 0,
        "trade_history": [],
        "nav_history": [],
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


def save_state(state: dict):
    """Persist state to JSON."""
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.info(f"State saved. NAV=${state['nav']:,.2f}  mode={state['mode']}")


def log_trade(trade: dict):
    """Append trade to JSONL log."""
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


# ── Data Fetching ──────────────────────────────────────────────────────────
def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetch adjusted close prices for tickers. Returns DataFrame with dates as index."""
    all_tickers = list(set(tickers + [BENCH, "^VIX"]))
    log.info(f"Fetching price data for {len(all_tickers)} tickers, period={period}...")

    data = yf.download(
        all_tickers,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns (Price, Ticker) in newer versions
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"]
    else:
        close = data[["Close"]].copy()
        volume = data[["Volume"]].copy()

    log.info(f"Got {len(close)} trading days, {close.shape[1]} tickers")
    return close, volume


def compute_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for the universe using T-1 data.
    Returns a DataFrame indexed by ticker with feature columns.
    """
    features = {}

    # Get VIX data
    vix_series = close["^VIX"] if "^VIX" in close.columns else None

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            log.warning(f"  {ticker}: no price data, skipping")
            continue

        px = close[ticker].dropna()
        vol_s = volume[ticker].dropna() if ticker in volume.columns else None

        if len(px) < 252:
            log.warning(f"  {ticker}: only {len(px)} days of data, need 252. Skipping.")
            continue

        # Use T-1 data (exclude today)
        px_t1 = px.iloc[:-1]
        last_px = px_t1.iloc[-1]

        # Returns
        ret_1d = px_t1.pct_change()
        ret_1m = px_t1.pct_change(21)
        ret_3m = px_t1.pct_change(63)

        # Volatility features
        vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
        vol_63d = ret_1d.rolling(63).std() * np.sqrt(252)
        vol_126d = ret_1d.rolling(126).std() * np.sqrt(252)

        # vol_20d percentile rank within its own history
        vol_20d_current = vol_20d.iloc[-1]
        vol_20d_pctrank = (vol_20d.dropna() <= vol_20d_current).mean()

        # vol_ratio = vol_20d / vol_63d (short-term vs long-term vol)
        vol_ratio = vol_20d.iloc[-1] / vol_63d.iloc[-1] if vol_63d.iloc[-1] > 0 else 1.0

        # Distance from 52-week high
        high_52w = px_t1.rolling(252).max().iloc[-1]
        dist_52w_high = (last_px - high_52w) / high_52w  # negative = below high

        # Distance from 200-day SMA (3m feature)
        sma_200 = px_t1.rolling(200).mean().iloc[-1]
        dist_200sma = (last_px - sma_200) / sma_200 if sma_200 > 0 else 0.0

        # Max drawdown over 52 weeks
        rolling_max = px_t1.iloc[-252:].cummax()
        drawdown = (px_t1.iloc[-252:] - rolling_max) / rolling_max
        max_dd_52w = drawdown.min()

        # Momentum acceleration (mom_1m change over past month)
        mom_1m_series = px_t1.pct_change(21)
        mom_accel = mom_1m_series.iloc[-1] - mom_1m_series.iloc[-22] if len(mom_1m_series) > 22 else 0.0

        # Sector return 1m (use own return as proxy — no sector mapping needed for ranking)
        sector_ret_1m = ret_1m.iloc[-1]

        # Volume surge
        if vol_s is not None and len(vol_s) > 20:
            vol_s_t1 = vol_s.iloc[:-1]
            avg_vol_20d = vol_s_t1.rolling(20).mean().iloc[-1]
            current_vol = vol_s_t1.iloc[-1]
            volume_surge = current_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0
        else:
            volume_surge = 1.0

        # VIX features
        vix_level = vix_series.iloc[-2] if vix_series is not None and len(vix_series) > 1 else 20.0

        features[ticker] = {
            "close": last_px,
            "vol_20d": vol_20d_current,
            "vol_63d": vol_63d.iloc[-1],
            "vol_126d": vol_126d.iloc[-1] if not pd.isna(vol_126d.iloc[-1]) else vol_63d.iloc[-1],
            "vol_20d_pctrank": vol_20d_pctrank,
            "vol_ratio": vol_ratio,
            "mom_1m": ret_1m.iloc[-1],
            "mom_3m": ret_3m.iloc[-1],
            "mom_accel": mom_accel,
            "dist_52w_high": dist_52w_high,
            "dist_200sma": dist_200sma,
            "max_dd_52w": max_dd_52w,
            "sector_ret_1m": sector_ret_1m,
            "volume_surge": volume_surge,
            "vix": vix_level,
        }

    df = pd.DataFrame(features).T
    log.info(f"Computed features for {len(df)} stocks")
    return df


# ── Asymmetric Filter ──────────────────────────────────────────────────────
def check_asymmetric_filter(features_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check if asymmetric filter fires.
    A stock passes if:
      - vol_20d > 80th percentile (within current universe cross-section)
      - mom_3m < -5%
      - volume_surge > 1.5x

    Filter fires if >= MIN_FILTER_STOCKS pass.
    Returns (fired: bool, passing_tickers: list).
    """
    df = features_df.copy()

    # 80th percentile of vol_20d within universe
    vol_threshold = df["vol_20d"].quantile(FILTER_VOL_PCTILE / 100.0)

    passing = df[
        (df["vol_20d"] > vol_threshold)
        & (df["mom_3m"] < FILTER_MOM_3M_THRESH)
        & (df["volume_surge"] > FILTER_VOLUME_SURGE)
    ]

    tickers = list(passing.index)
    fired = len(tickers) >= MIN_FILTER_STOCKS

    log.info(f"Asymmetric filter: {len(tickers)} stocks pass (threshold={MIN_FILTER_STOCKS})")
    log.info(f"  vol_20d > {vol_threshold:.4f}, mom_3m < {FILTER_MOM_3M_THRESH}, "
             f"volume_surge > {FILTER_VOLUME_SURGE}")
    if tickers:
        log.info(f"  Passing stocks: {tickers}")
    log.info(f"  Filter {'FIRES' if fired else 'does NOT fire'}")

    return fired, tickers


# ── Ranking / Scoring ──────────────────────────────────────────────────────
def rank_stocks(features_df: pd.DataFrame) -> pd.Series:
    """
    Rank stocks by composite distress z-score (1m + 3m combined).
    Higher score = more asymmetric upside potential.

    1m features (weight 0.4 each = 40% of total):
      z(vol_63d) + z(vix) + z(-mom_1m) + z(-dist_52w_high)

    3m features (weight 0.6 each = 60% of total):
      z(vol_126d) + z(-mom_3m) + z(-dist_200sma)

    The 3m features get higher weight because the fusion analysis showed
    3m-only (Sharpe 1.13) outperforms 1m-only (Sharpe 0.72) and is
    more regime-agnostic (bull 1.10, bear 1.36 vs 1m: bull 0.88, bear 0.37).
    """
    df = features_df.copy()

    def zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 0 else s * 0

    # 1m features (40% weight)
    w_1m = 0.4
    z_vol63 = zscore(df["vol_63d"])
    z_vix = zscore(df["vix"])
    z_neg_mom1m = zscore(-df["mom_1m"])
    z_neg_dist52w = zscore(-df["dist_52w_high"])
    score_1m = (z_vol63 + z_vix + z_neg_mom1m + z_neg_dist52w) / 4.0

    # 3m features (60% weight)
    w_3m = 0.6
    z_vol126 = zscore(df["vol_126d"])
    z_neg_mom3m = zscore(-df["mom_3m"])
    z_neg_dist200 = zscore(-df["dist_200sma"])
    score_3m = (z_vol126 + z_neg_mom3m + z_neg_dist200) / 3.0

    composite = w_1m * score_1m + w_3m * score_3m
    composite.name = "distress_score"

    ranked = composite.sort_values(ascending=False)
    log.info("Stock rankings (top 10) — 1m+3m combined scoring:")
    for i, (ticker, score) in enumerate(ranked.head(10).items()):
        log.info(f"  {i+1}. {ticker}: score={score:.3f}  "
                 f"vol_63d={df.loc[ticker, 'vol_63d']:.4f}  "
                 f"mom_1m={df.loc[ticker, 'mom_1m']:.4f}  "
                 f"mom_3m={df.loc[ticker, 'mom_3m']:.4f}  "
                 f"dist_200sma={df.loc[ticker, 'dist_200sma']:.4f}")

    return ranked


# ── Portfolio Operations ───────────────────────────────────────────────────
def mark_to_market(state: dict, close: pd.DataFrame) -> dict:
    """Update NAV based on current prices."""
    if not state["positions"]:
        state["nav"] = state["cash"]
        return state

    position_value = 0.0
    latest_prices = close.iloc[-1]

    for ticker, pos in state["positions"].items():
        if ticker in latest_prices and not pd.isna(latest_prices[ticker]):
            current_px = latest_prices[ticker]
            position_value += pos["shares"] * current_px
        else:
            # Use entry price as fallback
            position_value += pos["shares"] * pos["entry_price"]
            log.warning(f"  {ticker}: no current price, using entry price")

    state["nav"] = state["cash"] + position_value
    return state


def liquidate_positions(state: dict, close: pd.DataFrame, today_str: str) -> dict:
    """Sell all current positions at current prices."""
    if not state["positions"]:
        return state

    latest_prices = close.iloc[-1]

    for ticker, pos in list(state["positions"].items()):
        if ticker in latest_prices and not pd.isna(latest_prices[ticker]):
            sell_px = latest_prices[ticker]
        else:
            sell_px = pos["entry_price"]

        proceeds = pos["shares"] * sell_px
        cost = proceeds * (COST_BPS / 10000.0)  # half of round-trip on exit
        net_proceeds = proceeds - cost
        state["cash"] += net_proceeds

        pnl = (sell_px - pos["entry_price"]) * pos["shares"]
        pnl_pct = (sell_px / pos["entry_price"] - 1) * 100

        trade = {
            "date": today_str,
            "action": "SELL",
            "ticker": ticker,
            "shares": pos["shares"],
            "price": sell_px,
            "proceeds": net_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": "rebalance_exit",
        }
        log_trade(trade)
        state["trade_history"].append(trade)
        log.info(f"  SOLD {pos['shares']:.2f} {ticker} @ ${sell_px:.2f} "
                 f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

    state["positions"] = {}
    return state


def buy_positions(state: dict, tickers: list[str], close: pd.DataFrame, today_str: str) -> dict:
    """Buy equal-weight positions in given tickers."""
    if not tickers:
        return state

    latest_prices = close.iloc[-1]
    allocation_per_stock = state["cash"] / len(tickers)

    for ticker in tickers:
        if ticker not in latest_prices or pd.isna(latest_prices[ticker]):
            log.warning(f"  {ticker}: no price available, skipping")
            continue

        buy_px = latest_prices[ticker]
        cost = allocation_per_stock * (COST_BPS / 10000.0)  # half round-trip on entry
        investable = allocation_per_stock - cost
        shares = investable / buy_px

        state["positions"][ticker] = {
            "shares": shares,
            "entry_price": buy_px,
            "entry_date": today_str,
        }
        state["cash"] -= allocation_per_stock

        trade = {
            "date": today_str,
            "action": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": buy_px,
            "cost": allocation_per_stock,
            "reason": "rebalance_entry",
        }
        log_trade(trade)
        state["trade_history"].append(trade)
        log.info(f"  BOUGHT {shares:.2f} {ticker} @ ${buy_px:.2f} "
                 f"(${allocation_per_stock:,.2f} allocated)")

    return state


# ── Rebalance Logic ────────────────────────────────────────────────────────
def is_first_trading_day_of_month(today: date, close: pd.DataFrame) -> bool:
    """Check if today is the 1st trading day of the month."""
    # Get trading days from price data
    trading_days = close.index
    if hasattr(trading_days, 'date'):
        trading_dates = [d.date() if hasattr(d, 'date') else d for d in trading_days]
    else:
        trading_dates = list(trading_days)

    # Find trading days in current month
    current_month_days = [d for d in trading_dates
                          if d.year == today.year and d.month == today.month]

    if not current_month_days:
        return False

    first_trading_day = min(current_month_days)
    return today == first_trading_day


def run_rebalance(state: dict, close: pd.DataFrame, volume: pd.DataFrame,
                  today: date) -> dict:
    """Execute monthly rebalance."""
    today_str = today.isoformat()
    log.info(f"{'='*60}")
    log.info(f"REBALANCE DAY: {today_str}")
    log.info(f"{'='*60}")

    # 1. Compute features
    features_df = compute_features(close, volume)
    if features_df.empty:
        log.error("No features computed — aborting rebalance")
        return state

    # 2. Check asymmetric filter
    filter_fired, passing_tickers = check_asymmetric_filter(features_df)
    state["filter_fired"] = filter_fired
    state["filter_stocks_passing"] = len(passing_tickers)

    # 3. Liquidate current positions
    log.info("Liquidating current positions...")
    state = liquidate_positions(state, close, today_str)

    if filter_fired:
        # 4a. Filter fires: rank and buy top 5
        log.info("FILTER FIRED — ranking stocks for asymmetric picks")
        rankings = rank_stocks(features_df)
        top_picks = list(rankings.head(TOP_K).index)
        log.info(f"Top {TOP_K} picks: {top_picks}")
        state = buy_positions(state, top_picks, close, today_str)
        state["mode"] = "PICKS"
    else:
        # 4b. Filter doesn't fire: hold SPY
        log.info("FILTER DID NOT FIRE — holding SPY")
        state = buy_positions(state, [BENCH], close, today_str)
        state["mode"] = "SPY"

    state["last_rebalance_date"] = today_str

    # Update NAV
    state = mark_to_market(state, close)

    # Record NAV snapshot
    state["nav_history"].append({
        "date": today_str,
        "nav": state["nav"],
        "mode": state["mode"],
        "filter_fired": filter_fired,
        "positions": list(state["positions"].keys()),
    })

    log.info(f"Post-rebalance NAV: ${state['nav']:,.2f}  mode={state['mode']}")
    return state


# ── Main Entry Point ───────────────────────────────────────────────────────
def main(force_rebalance: bool = False, check_only: bool = False):
    """
    Main entry point. Called by cron or manually.

    Args:
        force_rebalance: Force rebalance regardless of date.
        check_only: Just check filter status and mark-to-market, don't trade.
    """
    log.info("=" * 70)
    log.info("ML Asymmetric Stock Ranker — Paper Trading Engine")
    log.info(f"Run at: {datetime.now().isoformat()}")
    log.info("=" * 70)

    state = load_state()
    today = date.today()
    today_str = today.isoformat()

    log.info(f"Current state: NAV=${state['nav']:,.2f}  mode={state['mode']}  "
             f"positions={list(state['positions'].keys())}")
    log.info(f"Last rebalance: {state['last_rebalance_date']}")

    # Fetch price data
    try:
        close, volume = fetch_prices(UNIVERSE, period="18mo")
    except Exception as e:
        log.error(f"Failed to fetch prices: {e}")
        log.error(traceback.format_exc())
        save_state(state)
        return state

    if close.empty:
        log.error("No price data returned — aborting")
        save_state(state)
        return state

    if check_only:
        log.info("CHECK-ONLY mode — computing features and filter status")
        features_df = compute_features(close, volume)
        if not features_df.empty:
            filter_fired, passing = check_asymmetric_filter(features_df)
            rankings = rank_stocks(features_df)
            log.info(f"\nFilter status: {'FIRES' if filter_fired else 'DOES NOT FIRE'}")
            log.info(f"Stocks passing filter: {len(passing)}")
            if passing:
                log.info(f"Passing: {passing}")

        # Mark-to-market
        state = mark_to_market(state, close)
        save_state(state)
        log.info(f"Current NAV: ${state['nav']:,.2f}")
        return state

    # Check if it's rebalance day
    is_rebal_day = is_first_trading_day_of_month(today, close) or force_rebalance

    if is_rebal_day:
        log.info("Today is a rebalance day")
        state = run_rebalance(state, close, volume, today)
    else:
        log.info("Not a rebalance day — mark-to-market only")
        state = mark_to_market(state, close)

        # Record daily NAV
        state["nav_history"].append({
            "date": today_str,
            "nav": state["nav"],
            "mode": state["mode"],
            "filter_fired": state.get("filter_fired", False),
            "positions": list(state["positions"].keys()),
        })

    save_state(state)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY")
    log.info(f"  NAV:       ${state['nav']:,.2f}")
    log.info(f"  Cash:      ${state['cash']:,.2f}")
    log.info(f"  Mode:      {state['mode']}")
    log.info(f"  Positions: {list(state['positions'].keys()) or 'none'}")
    log.info(f"  Trades:    {len(state['trade_history'])} total")
    log.info(f"{'='*60}")

    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Stock Ranker Paper Engine")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force rebalance regardless of date")
    parser.add_argument("--check-only", action="store_true",
                        help="Check filter status without trading")
    args = parser.parse_args()

    main(force_rebalance=args.force_rebalance, check_only=args.check_only)
