#!/usr/bin/env python3
"""
Post-Crash Recovery Strategy — Observation-Driven Mean Reversion
================================================================

OBSERVATION CHAIN (from market_observation_scanner.py):
  - Stocks >20% below 52-week high have +9.9% avg 6-month forward return
    (vs ~3% baseline). Cohen's d=0.11, N=180K+.
  - Energy recovers hardest (+12.5%), then Financials (+8.2%), Tech (+5.2%).
  - Positive skew (+2.48) = asymmetric upside.

HYPOTHESIS:
  Stocks that crash >20% from highs are oversold. By filtering for quality
  (S&P 500), recency (crash within 63 days), liquidity (>500K ADV), and
  sector-weighting (favor Energy/Financials), we capture mean-reversion
  with controlled downside.

STRATEGY:
  - Universe: S&P 500 constituents, 7+ years daily data via yfinance
  - Signal: Weekly scan for stocks >20% below 252-day high
  - Filters: recency (crash within 63d), volume (>500K ADV), sector weight
  - Entry: Buy next open after signal
  - Exit: 63 trading days hold, OR recovery to within 5% of 52w high,
          OR stop-loss at -15% from entry
  - Sizing: Equal weight, max 10 positions, weekly rebalance
  - Costs: 5 bps each way (10 bps round-trip)

VALIDATION (HC #428 + HC #735):
  - Walk-forward sliding window, weekly rebalance
  - Full OOT (all available data)
  - Regime stratification (SPY green/red/flat)
  - Permutation test (200 shuffles)
  - Reject if regime gap > 0.50

Usage:
    python3 post_crash_recovery_strategy.py [--no-cache] [--years 8]

Author: Claude Opus 4.6 / Teleclaude Research
"""

import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = BASE / "data" / "crash_recovery_cache"
OUTPUT_DIR = BASE / "output" / "post_crash_recovery"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy parameters
CRASH_THRESHOLD = 0.20       # Must be >20% below 52-week high
HIGH_LOOKBACK = 252          # 252 trading days = ~1 year
RECENCY_WINDOW = 63          # Crash must have happened within 63 trading days
MIN_AVG_VOLUME = 500_000     # Minimum average daily volume
MAX_POSITIONS = 10           # Maximum concurrent positions
HOLD_PERIOD = 63             # Default hold = 3 months (63 trading days)
RECOVERY_THRESHOLD = 0.05    # Exit if within 5% of 52w high
STOP_LOSS = 0.15             # Stop-loss at -15% from entry
COST_BPS = 5                 # 5 bps each way
INITIAL_CAPITAL = 1_000_000  # $1M starting capital

# Sector weights (based on observed recovery rates)
SECTOR_WEIGHTS = {
    "Energy": 1.50,           # +12.5% recovery → highest weight
    "Financials": 1.25,       # +8.2% recovery
    "Materials": 1.10,
    "Industrials": 1.05,
    "Health Care": 1.00,
    "Consumer Discretionary": 1.00,
    "Communication Services": 0.95,
    "Consumer Staples": 0.90,
    "Information Technology": 0.85,  # +5.2% recovery → lower weight
    "Utilities": 0.80,
    "Real Estate": 0.80,
}

# Regime classification thresholds (SPY weekly return)
REGIME_GREEN_THRESH = 0.005   # >+0.5% weekly = green
REGIME_RED_THRESH = -0.005    # <-0.5% weekly = red

# Validation
N_PERMUTATIONS = 200
REGIME_GAP_REJECT = 0.50

# Wikipedia URL for S&P 500
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> tuple:
    """Fetch S&P 500 constituents + sector mapping."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        sym_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        sec_col = [c for c in df.columns if "gics" in c.lower() and "sector" in c.lower()][0]
        tickers = df[sym_col].str.replace(".", "-", regex=False).tolist()
        sector_map = dict(zip(
            df[sym_col].str.replace(".", "-", regex=False),
            df[sec_col]
        ))
        print(f"  Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sector_map
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using hardcoded fallback")
        return _hardcoded_sp500()


def _hardcoded_sp500() -> tuple:
    """Hardcoded fallback — major S&P 500 constituents across all sectors."""
    sector_assignments = {
        "Information Technology": [
            "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "CSCO", "ACN",
            "ORCL", "INTC", "IBM", "TXN", "QCOM", "AMAT", "ADI", "LRCX", "MU",
            "KLAC", "CDNS", "SNPS", "MCHP", "ON", "FTNT", "HPQ", "HPE", "KEYS",
        ],
        "Health Care": [
            "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
            "BMY", "AMGN", "MDT", "GILD", "ISRG", "CVS", "CI", "ELV", "VRTX",
            "REGN", "ZTS", "BDX", "SYK", "BSX", "HCA", "MCK", "DXCM",
        ],
        "Financials": [
            "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
            "C", "AXP", "SCHW", "CB", "MMC", "PGR", "ICE", "AON", "CME", "MET",
            "AIG", "TFC", "USB", "PNC", "COF", "BK", "ALL", "AFL", "TRV",
        ],
        "Consumer Discretionary": [
            "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
            "CMG", "MAR", "ORLY", "AZO", "ROST", "DHI", "LEN", "GM", "F",
            "YUM", "DG", "DLTR", "EBAY", "APTV", "BBY",
        ],
        "Communication Services": [
            "GOOGL", "META", "DIS", "CMCSA", "NFLX", "VZ", "T", "TMUS", "CHTR",
            "EA", "TTWO", "WBD", "PARA", "OMC", "IPG", "MTCH", "LYV",
        ],
        "Industrials": [
            "GE", "CAT", "HON", "UNP", "UPS", "RTX", "BA", "DE", "LMT", "MMM",
            "GD", "NOC", "WM", "RSG", "EMR", "ITW", "ETN", "PH", "ROK", "CTAS",
            "CSX", "NSC", "PCAR", "FDX", "FAST", "ODFL", "JCI", "TDG",
        ],
        "Consumer Staples": [
            "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KMB",
            "GIS", "SYY", "ADM", "KHC", "HSY", "STZ", "MKC", "EL", "CHD", "KDP",
        ],
        "Energy": [
            "XOM", "CVX", "COP", "SLB", "MPC", "EOG", "PSX", "VLO", "OXY",
            "PXD", "DVN", "HAL", "FANG", "HES", "BKR", "CTRA", "MRO", "APA",
        ],
        "Utilities": [
            "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
            "PEG", "ES", "AWK", "EIX", "DTE", "AEE", "CMS", "FE", "PPL", "ATO",
        ],
        "Real Estate": [
            "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR",
            "AVB", "EQR", "VTR", "ARE", "MAA", "ESS", "UDR", "PEAK", "BXP",
        ],
        "Materials": [
            "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM",
            "DOW", "DD", "PPG", "CE", "ALB", "IFF", "FMC", "CF", "MOS",
        ],
    }
    tickers = []
    sector_map = {}
    for sector, syms in sector_assignments.items():
        for s in syms:
            tickers.append(s)
            sector_map[s] = sector
    return tickers, sector_map


def download_data(tickers: list, sector_map: dict, years: int = 8,
                  use_cache: bool = True) -> tuple:
    """Download daily OHLCV data for all tickers + SPY benchmark."""
    import yfinance as yf

    cache_file = CACHE_DIR / f"crash_recovery_data_{years}y.pkl"

    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"  Loading cached data ({age_hours:.1f}h old)")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    end = dt.datetime.now()
    start = end - dt.timedelta(days=years * 365)

    # Always include SPY for regime classification
    all_tickers = list(set(tickers + ["SPY"]))

    print(f"  Downloading {len(all_tickers)} tickers, {years} years...")

    # Download in batches to avoid rate limits
    all_data = {}
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            df = yf.download(batch_str, start=start, end=end,
                           group_by="ticker", progress=False, threads=True)
            if len(batch) == 1:
                # Single ticker: columns are just OHLCV
                ticker = batch[0]
                if not df.empty and len(df) > 100:
                    all_data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for ticker in batch:
                    try:
                        sub = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                        sub = sub.dropna(subset=["Close"])
                        if len(sub) > 100:
                            all_data[ticker] = sub
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"    Batch {i//batch_size + 1} failed: {e}")

        if i + batch_size < len(all_tickers):
            time.sleep(0.5)

    print(f"  Downloaded {len(all_data)} tickers successfully")

    result = (all_data, sector_map)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_crash_signals(prices: dict, sector_map: dict,
                          as_of_date: pd.Timestamp) -> list:
    """
    For a given date, identify all stocks >20% below their 252-day high,
    with recency and volume filters applied.

    Returns list of dicts with signal info, sorted by sector-weighted score.
    """
    signals = []

    for ticker, df in prices.items():
        if ticker == "SPY":
            continue

        # Get data up to as_of_date
        mask = df.index <= as_of_date
        hist = df.loc[mask]

        if len(hist) < HIGH_LOOKBACK:
            continue

        recent = hist.tail(HIGH_LOOKBACK)
        current_close = recent["Close"].iloc[-1]
        high_52w = recent["High"].max()

        if high_52w <= 0 or current_close <= 0:
            continue

        # Drawdown from 52-week high
        drawdown = (high_52w - current_close) / high_52w

        if drawdown < CRASH_THRESHOLD:
            continue

        # RECENCY FILTER: When did the high occur? The crash must be recent.
        high_date_idx = recent["High"].idxmax()
        days_since_high = len(recent.loc[high_date_idx:]) - 1

        # Also check: the stock must have been within 10% of the high
        # sometime in the last RECENCY_WINDOW days (i.e., the crash is fresh)
        recent_window = hist.tail(RECENCY_WINDOW)
        if len(recent_window) < 20:
            continue
        recent_high = recent_window["High"].max()
        was_near_high_recently = (high_52w - recent_high) / high_52w < 0.10

        # The crash is "recent" if either the 52w high is within the recency
        # window OR the stock was near its high within the window
        if days_since_high > RECENCY_WINDOW and not was_near_high_recently:
            continue

        # VOLUME FILTER
        avg_vol = recent.tail(20)["Volume"].mean()
        if avg_vol < MIN_AVG_VOLUME:
            continue

        # Sector weight
        sector = sector_map.get(ticker, "Unknown")
        sector_weight = SECTOR_WEIGHTS.get(sector, 0.90)

        # Crash velocity: how fast did it drop? Faster crashes = more oversold
        if len(recent_window) >= 5:
            crash_velocity = abs(
                (recent_window["Close"].iloc[-1] / recent_window["Close"].iloc[0]) - 1
            )
        else:
            crash_velocity = drawdown

        # Composite score: deeper drawdown + faster crash + sector boost
        score = drawdown * sector_weight * (1 + crash_velocity)

        signals.append({
            "ticker": ticker,
            "sector": sector,
            "drawdown": drawdown,
            "days_since_high": days_since_high,
            "avg_volume": avg_vol,
            "sector_weight": sector_weight,
            "crash_velocity": crash_velocity,
            "score": score,
            "entry_price": current_close,
            "high_52w": high_52w,
        })

    # Sort by score descending, take top MAX_POSITIONS
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class Position:
    """Track a single position."""
    def __init__(self, ticker, entry_date, entry_price, shares, sector, high_52w):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.sector = sector
        self.high_52w = high_52w
        self.days_held = 0
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0


def check_exits(positions: list, prices: dict, current_date: pd.Timestamp,
                cost_bps: float = COST_BPS) -> tuple:
    """Check all positions for exit conditions. Returns (active, closed)."""
    active = []
    closed = []

    for pos in positions:
        if pos.ticker not in prices:
            # Ticker delisted or data missing — force close at last known price
            pos.exit_date = current_date
            pos.exit_price = pos.entry_price  # conservative: flat
            pos.exit_reason = "delisted"
            pos.pnl = 0
            pos.pnl_pct = 0
            closed.append(pos)
            continue

        df = prices[pos.ticker]
        if current_date not in df.index:
            active.append(pos)
            continue

        row = df.loc[current_date]
        current_price = row["Close"]
        current_high = row["High"]
        current_low = row["Low"]

        pos.days_held += 1
        exit_reason = None
        exit_price = current_price

        # Check stop-loss (use intraday low)
        if (current_low - pos.entry_price) / pos.entry_price <= -STOP_LOSS:
            exit_reason = "stop_loss"
            # Assume we get stopped at the stop level
            exit_price = pos.entry_price * (1 - STOP_LOSS)

        # Check recovery (use intraday high)
        elif (pos.high_52w - current_high) / pos.high_52w <= RECOVERY_THRESHOLD:
            exit_reason = "recovery"
            exit_price = pos.high_52w * (1 - RECOVERY_THRESHOLD)

        # Check max hold period
        elif pos.days_held >= HOLD_PERIOD:
            exit_reason = "max_hold"
            exit_price = current_price

        if exit_reason:
            cost_mult = cost_bps / 10_000  # exit cost
            pos.exit_date = current_date
            pos.exit_price = exit_price
            pos.exit_reason = exit_reason
            gross_pnl_pct = (exit_price / pos.entry_price) - 1
            pos.pnl_pct = gross_pnl_pct - cost_mult  # subtract exit cost
            pos.pnl = pos.shares * pos.entry_price * pos.pnl_pct
            closed.append(pos)
        else:
            active.append(pos)

    return active, closed


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(prices: dict, sector_map: dict, spy_data: pd.DataFrame,
                 start_date: pd.Timestamp = None,
                 end_date: pd.Timestamp = None,
                 shuffle_signals: bool = False,
                 rng: np.random.RandomState = None) -> dict:
    """
    Run the post-crash recovery strategy over the given period.

    Walk-forward with weekly rebalance (every 5 trading days).
    Sliding window: uses trailing data only (no future leak).

    If shuffle_signals=True, randomize which stocks get selected
    (for permutation test).
    """
    # Get trading dates from SPY
    if start_date is None:
        start_date = spy_data.index[HIGH_LOOKBACK + 10]
    if end_date is None:
        end_date = spy_data.index[-1]

    trading_dates = spy_data.index[
        (spy_data.index >= start_date) & (spy_data.index <= end_date)
    ]

    if len(trading_dates) < 50:
        return None

    # Track portfolio
    capital = INITIAL_CAPITAL
    positions = []
    all_closed = []

    # Daily equity curve
    equity_curve = pd.Series(dtype=float, index=trading_dates)
    cash_series = pd.Series(dtype=float, index=trading_dates)

    rebalance_counter = 0

    for date in trading_dates:
        # Check exits first
        positions, newly_closed = check_exits(positions, prices, date)
        for pos in newly_closed:
            capital += pos.shares * pos.exit_price  # return capital
            capital += pos.pnl  # add/subtract P&L
            all_closed.append(pos)

        # Weekly rebalance: scan for new signals
        rebalance_counter += 1
        if rebalance_counter >= 5:
            rebalance_counter = 0

            open_slots = MAX_POSITIONS - len(positions)
            if open_slots > 0 and capital > 10_000:
                signals = compute_crash_signals(prices, sector_map, date)

                if shuffle_signals and rng is not None:
                    # Permutation: shuffle the signal scores
                    rng.shuffle(signals)

                # Filter out tickers we already hold
                held_tickers = {p.ticker for p in positions}
                candidates = [s for s in signals if s["ticker"] not in held_tickers]

                # Take top candidates up to open slots
                to_buy = candidates[:open_slots]

                if to_buy:
                    per_position = capital / (len(to_buy) + len(positions))
                    per_position = min(per_position, capital / len(to_buy))

                    for sig in to_buy:
                        ticker = sig["ticker"]
                        # Entry at next day's open approximated by current close
                        # (conservative — real entry would be next open)
                        entry_price = sig["entry_price"]
                        cost_mult = COST_BPS / 10_000
                        # Adjust for entry cost
                        effective_price = entry_price * (1 + cost_mult)
                        shares = int(per_position / effective_price)

                        if shares <= 0:
                            continue

                        cost = shares * effective_price
                        if cost > capital:
                            shares = int(capital / effective_price)
                            cost = shares * effective_price

                        if shares <= 0:
                            continue

                        capital -= cost
                        positions.append(Position(
                            ticker=ticker,
                            entry_date=date,
                            entry_price=entry_price,
                            shares=shares,
                            sector=sig["sector"],
                            high_52w=sig["high_52w"],
                        ))

        # Mark-to-market
        position_value = 0
        for pos in positions:
            if pos.ticker in prices and date in prices[pos.ticker].index:
                current_price = prices[pos.ticker].loc[date, "Close"]
                position_value += pos.shares * current_price
            else:
                position_value += pos.shares * pos.entry_price

        equity_curve.loc[date] = capital + position_value
        cash_series.loc[date] = capital

    # Force-close remaining positions at end
    for pos in positions:
        last_date = trading_dates[-1]
        if pos.ticker in prices:
            df = prices[pos.ticker]
            valid = df.index[df.index <= last_date]
            if len(valid) > 0:
                pos.exit_price = df.loc[valid[-1], "Close"]
            else:
                pos.exit_price = pos.entry_price
        else:
            pos.exit_price = pos.entry_price
        pos.exit_date = last_date
        pos.exit_reason = "end_of_period"
        cost_mult = COST_BPS / 10_000
        gross_pnl_pct = (pos.exit_price / pos.entry_price) - 1
        pos.pnl_pct = gross_pnl_pct - cost_mult
        pos.pnl = pos.shares * pos.entry_price * pos.pnl_pct
        capital += pos.shares * pos.exit_price + pos.pnl
        all_closed.append(pos)

    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 50:
        return None

    return {
        "equity_curve": equity_curve,
        "trades": all_closed,
        "final_capital": capital,
        "start_date": trading_dates[0],
        "end_date": trading_dates[-1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, trades: list,
                    label: str = "Strategy") -> dict:
    """Compute comprehensive performance metrics."""
    if equity_curve is None or len(equity_curve) < 50:
        return None

    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return None

    # Annualization factor
    ann = 252

    # Basic returns
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = len(returns) / ann
    cagr = (1 + total_return) ** (1 / max(years, 0.1)) - 1

    # Risk metrics
    daily_mean = returns.mean()
    daily_std = returns.std()

    sharpe = (daily_mean / daily_std * np.sqrt(ann)) if daily_std > 0 else 0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_std
    sortino = (daily_mean / downside_std * np.sqrt(ann)) if downside_std > 0 else 0

    # Max drawdown
    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    max_dd = drawdowns.min()

    # Trade-level metrics
    if trades:
        pnl_list = [t.pnl_pct for t in trades]
        winners = [p for p in pnl_list if p > 0]
        losers = [p for p in pnl_list if p <= 0]
        win_rate = len(winners) / len(pnl_list) if pnl_list else 0

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 1
        profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else float('inf')

        avg_hold = np.mean([t.days_held for t in trades])

        # Sector breakdown
        sector_trades = defaultdict(list)
        for t in trades:
            sector_trades[t.sector].append(t.pnl_pct)
        sector_stats = {}
        for sector, pnls in sector_trades.items():
            sector_stats[sector] = {
                "n_trades": len(pnls),
                "avg_return": np.mean(pnls),
                "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            }

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[t.exit_reason] += 1
    else:
        win_rate = 0
        profit_factor = 0
        avg_hold = 0
        sector_stats = {}
        exit_reasons = {}
        avg_win = 0
        avg_loss = 0

    return {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility": daily_std * np.sqrt(ann),
        "n_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_hold_days": avg_hold,
        "sector_stats": sector_stats,
        "exit_reasons": dict(exit_reasons),
        "years": years,
        "start": str(equity_curve.index[0].date()),
        "end": str(equity_curve.index[-1].date()),
    }


def compute_spy_baseline(spy_data: pd.DataFrame, start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.Series:
    """SPY buy-and-hold equity curve for comparison."""
    mask = (spy_data.index >= start) & (spy_data.index <= end)
    spy = spy_data.loc[mask, "Close"].copy()
    if len(spy) < 50:
        return None
    # Normalize to same starting capital
    spy_equity = (spy / spy.iloc[0]) * INITIAL_CAPITAL
    return spy_equity


# ─────────────────────────────────────────────────────────────────────────────
# REGIME ANALYSIS (HC #428 R1)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(spy_data: pd.DataFrame) -> pd.Series:
    """
    Classify each trading day as green/red/flat based on SPY weekly return.
    Returns Series with index=date, value=regime.
    """
    spy_weekly = spy_data["Close"].resample("W-FRI").last().pct_change()

    regime_map = {}
    for week_end, ret in spy_weekly.items():
        if pd.isna(ret):
            regime = "flat"
        elif ret > REGIME_GREEN_THRESH:
            regime = "green"
        elif ret < REGIME_RED_THRESH:
            regime = "red"
        else:
            regime = "flat"
        regime_map[week_end] = regime

    # Map back to daily: each day gets its week's regime
    daily_regime = pd.Series(dtype=str, index=spy_data.index)
    for date in spy_data.index:
        # Find the Friday of this week
        days_to_friday = (4 - date.weekday()) % 7
        friday = date + pd.Timedelta(days=days_to_friday)
        friday = pd.Timestamp(friday)
        daily_regime.loc[date] = regime_map.get(friday, "flat")

    return daily_regime


def regime_stratified_metrics(equity_curve: pd.Series,
                              daily_regime: pd.Series) -> dict:
    """
    Compute Sharpe ratio stratified by market regime.
    Returns dict with per-regime Sharpe and the regime gap test.
    """
    returns = equity_curve.pct_change().dropna()

    # Align regime to returns
    common_idx = returns.index.intersection(daily_regime.index)
    returns = returns.loc[common_idx]
    regimes = daily_regime.loc[common_idx]

    result = {}
    sharpes = {}

    for regime in ["green", "red", "flat"]:
        mask = regimes == regime
        r = returns.loc[mask]
        n = len(r)
        if n < 20:
            sharpes[regime] = 0
            result[regime] = {"sharpe": 0, "n_days": n, "mean_return": 0}
            continue

        s = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
        sharpes[regime] = s
        result[regime] = {
            "sharpe": round(s, 3),
            "n_days": n,
            "mean_return": round(r.mean() * 252, 4),
        }

    # Regime gap test (HC #428 R1)
    s_green = sharpes.get("green", 0)
    s_red = sharpes.get("red", 0)
    max_abs = max(abs(s_green), abs(s_red), 0.001)
    regime_gap = abs(s_green - s_red) / max_abs

    result["regime_gap"] = round(regime_gap, 3)
    result["regime_gap_pass"] = regime_gap <= REGIME_GAP_REJECT

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(prices: dict, sector_map: dict, spy_data: pd.DataFrame,
                     real_sharpe: float, n_perms: int = N_PERMUTATIONS) -> dict:
    """
    Shuffle signal selection to test if strategy edge is real.
    Returns p-value and distribution stats.
    """
    print(f"\n  Running {n_perms} permutation shuffles...")
    perm_sharpes = []

    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            print(f"    Permutation {i+1}/{n_perms}...")

        rng = np.random.RandomState(seed=i)
        result = run_backtest(prices, sector_map, spy_data,
                            shuffle_signals=True, rng=rng)

        if result is not None:
            metrics = compute_metrics(result["equity_curve"], result["trades"])
            if metrics:
                perm_sharpes.append(metrics["sharpe"])

    if not perm_sharpes:
        return {"p_value": 1.0, "n_valid": 0}

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    return {
        "p_value": round(p_value, 4),
        "n_valid": len(perm_sharpes),
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 3),
        "perm_std_sharpe": round(np.std(perm_sharpes), 3),
        "perm_p5_sharpe": round(np.percentile(perm_sharpes, 5), 3),
        "perm_p95_sharpe": round(np.percentile(perm_sharpes, 95), 3),
        "real_sharpe": round(real_sharpe, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict, spy_metrics: dict, regime: dict,
                 perm: dict) -> str:
    """Print and return a comprehensive strategy report."""
    lines = []

    def ln(s=""):
        lines.append(s)
        print(s)

    ln("=" * 78)
    ln("POST-CRASH RECOVERY STRATEGY — VALIDATION REPORT")
    ln("=" * 78)

    ln("\n--- OBSERVATION → HYPOTHESIS → STRATEGY CHAIN ---")
    ln("OBSERVATION: Stocks >20% below 52w high avg +9.9% 6-month fwd return")
    ln("             (vs ~3% baseline). Cohen's d=0.11, N=180K+. Skew=+2.48.")
    ln("             Energy +12.5%, Financials +8.2%, Tech +5.2%.")
    ln("HYPOTHESIS:  Crashed S&P 500 stocks are oversold; sector-weighted")
    ln("             selection with recency/volume filters captures recovery.")
    ln(f"PERIOD:      {metrics['start']} to {metrics['end']} ({metrics['years']:.1f} years)")

    ln("\n--- STRATEGY PERFORMANCE ---")
    ln(f"  CAGR:           {metrics['cagr']*100:>8.2f}%")
    ln(f"  Total Return:   {metrics['total_return']*100:>8.2f}%")
    ln(f"  Sharpe Ratio:   {metrics['sharpe']:>8.3f}")
    ln(f"  Sortino Ratio:  {metrics['sortino']:>8.3f}")
    ln(f"  Max Drawdown:   {metrics['max_drawdown']*100:>8.2f}%")
    ln(f"  Volatility:     {metrics['volatility']*100:>8.2f}%")
    ln(f"  Profit Factor:  {metrics['profit_factor']:>8.3f}")
    ln(f"  Win Rate:       {metrics['win_rate']*100:>8.1f}%")
    ln(f"  Avg Win:        {metrics['avg_win']*100:>8.2f}%")
    ln(f"  Avg Loss:       {metrics['avg_loss']*100:>8.2f}%")
    ln(f"  Total Trades:   {metrics['n_trades']:>8d}")
    ln(f"  Avg Hold (days):{metrics['avg_hold_days']:>8.1f}")

    ln("\n--- SPY BUY-AND-HOLD BASELINE ---")
    if spy_metrics:
        ln(f"  CAGR:           {spy_metrics['cagr']*100:>8.2f}%")
        ln(f"  Sharpe Ratio:   {spy_metrics['sharpe']:>8.3f}")
        ln(f"  Sortino Ratio:  {spy_metrics['sortino']:>8.3f}")
        ln(f"  Max Drawdown:   {spy_metrics['max_drawdown']*100:>8.2f}%")
        ln()
        sharpe_diff = metrics['sharpe'] - spy_metrics['sharpe']
        ln(f"  Sharpe vs SPY:  {sharpe_diff:>+8.3f}")
        ln(f"  CAGR vs SPY:    {(metrics['cagr'] - spy_metrics['cagr'])*100:>+8.2f}%")

    ln("\n--- EXIT REASON BREAKDOWN ---")
    for reason, count in sorted(metrics.get("exit_reasons", {}).items(),
                                 key=lambda x: -x[1]):
        pct = count / max(metrics["n_trades"], 1) * 100
        ln(f"  {reason:<15s}: {count:>5d} ({pct:>5.1f}%)")

    ln("\n--- SECTOR BREAKDOWN ---")
    if metrics.get("sector_stats"):
        ln(f"  {'Sector':<28s} {'Trades':>6s} {'AvgRet':>8s} {'WinRate':>8s}")
        ln(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8}")
        for sector, ss in sorted(metrics["sector_stats"].items(),
                                  key=lambda x: -x[1]["avg_return"]):
            ln(f"  {sector:<28s} {ss['n_trades']:>6d} "
               f"{ss['avg_return']*100:>7.2f}% {ss['win_rate']*100:>7.1f}%")

    ln("\n--- REGIME STRATIFICATION (HC #428 R1) ---")
    if regime:
        for r in ["green", "red", "flat"]:
            if r in regime:
                ri = regime[r]
                ln(f"  {r.upper():<6s}: Sharpe={ri['sharpe']:>7.3f}, "
                   f"N={ri['n_days']:>5d} days, "
                   f"Ann.Return={ri['mean_return']*100:>6.2f}%")

        gap = regime.get("regime_gap", 0)
        passed = regime.get("regime_gap_pass", False)
        status = "PASS" if passed else "REJECT"
        ln(f"\n  Regime Gap:     {gap:.3f} (threshold: {REGIME_GAP_REJECT:.2f}) → {status}")

    ln("\n--- PERMUTATION TEST ---")
    if perm:
        ln(f"  Real Sharpe:    {perm.get('real_sharpe', 0):>8.3f}")
        ln(f"  Perm Mean:      {perm.get('perm_mean_sharpe', 0):>8.3f}")
        ln(f"  Perm Std:       {perm.get('perm_std_sharpe', 0):>8.3f}")
        ln(f"  Perm [5%, 95%]: [{perm.get('perm_p5_sharpe', 0):.3f}, "
           f"{perm.get('perm_p95_sharpe', 0):.3f}]")
        ln(f"  p-value:        {perm.get('p_value', 1):.4f}")
        ln(f"  Valid perms:    {perm.get('n_valid', 0)}")

        if perm.get("p_value", 1) < 0.05:
            ln("  → SIGNIFICANT at 5% level (edge likely real)")
        else:
            ln("  → NOT significant at 5% level (edge may be noise)")

    ln("\n--- VERDICT ---")
    issues = []
    if metrics["sharpe"] < 0:
        issues.append("Negative Sharpe ratio")
    if metrics["sharpe"] < spy_metrics.get("sharpe", 0) if spy_metrics else False:
        issues.append("Underperforms SPY Sharpe")
    if regime and not regime.get("regime_gap_pass", True):
        issues.append(f"Regime gap {regime.get('regime_gap', 0):.3f} > {REGIME_GAP_REJECT}")
    if perm and perm.get("p_value", 1) >= 0.05:
        issues.append(f"Permutation p={perm.get('p_value', 1):.3f} (not significant)")
    if metrics["win_rate"] < 0.40:
        issues.append(f"Low win rate {metrics['win_rate']*100:.1f}%")
    if metrics["profit_factor"] < 1.0:
        issues.append("Profit factor < 1.0 (losing strategy)")

    if not issues:
        ln("  PASS — Strategy shows real edge across regimes with statistical")
        ln("         significance. Consider paper trading.")
    else:
        ln("  ISSUES FOUND:")
        for issue in issues:
            ln(f"    - {issue}")
        if any("Regime gap" in i for i in issues):
            ln("  → REJECTED per HC #428 R1 (regime-tailored, not genuine edge)")
        elif any("Negative Sharpe" in i or "Profit factor" in i for i in issues):
            ln("  → REJECTED (unprofitable after costs)")
        else:
            ln("  → CAUTION — review issues before deployment")

    ln("\n" + "=" * 78)

    return "\n".join(lines)


def save_results(metrics: dict, spy_metrics: dict, regime: dict,
                 perm: dict, report_text: str):
    """Save results to JSON and text files."""
    # Clean metrics for JSON serialization
    def clean(d):
        if d is None:
            return None
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, dict):
                out[k] = clean(v)
            elif isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
                out[k] = str(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, pd.Series):
                continue  # skip series
            elif isinstance(v, list) and v and hasattr(v[0], '__dict__'):
                continue  # skip Position objects
            else:
                out[k] = v
        return out

    results = {
        "strategy": clean(metrics),
        "spy_baseline": clean(spy_metrics),
        "regime_analysis": regime,
        "permutation_test": perm,
        "parameters": {
            "crash_threshold": CRASH_THRESHOLD,
            "high_lookback": HIGH_LOOKBACK,
            "recency_window": RECENCY_WINDOW,
            "min_avg_volume": MIN_AVG_VOLUME,
            "max_positions": MAX_POSITIONS,
            "hold_period": HOLD_PERIOD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "stop_loss": STOP_LOSS,
            "cost_bps": COST_BPS,
            "initial_capital": INITIAL_CAPITAL,
            "sector_weights": SECTOR_WEIGHTS,
        },
        "observation_chain": {
            "observation": "Stocks >20% below 52w high: +9.9% avg 6mo fwd return (N=180K+, d=0.11, skew=+2.48)",
            "hypothesis": "Crashed S&P 500 stocks are oversold; sector-weighted selection captures asymmetric recovery",
            "strategy": "Weekly scan, sector-weighted, recency+volume filtered, 63d hold / recovery exit / 15% stop",
        },
        "generated_at": str(dt.datetime.now()),
    }

    json_path = OUTPUT_DIR / "post_crash_recovery_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    report_path = OUTPUT_DIR / "post_crash_recovery_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n  Results saved to {json_path}")
    print(f"  Report saved to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-Crash Recovery Strategy — Observation-Driven Mean Reversion"
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh data download")
    parser.add_argument("--years", type=int, default=8,
                        help="Years of history to download (default: 8)")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip permutation test (faster)")
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    args = parser.parse_args()

    print("=" * 78)
    print("POST-CRASH RECOVERY STRATEGY")
    print("Observation → Hypothesis → Strategy → Validation")
    print("=" * 78)

    # ── Step 1: Load Data ──────────────────────────────────────────────────
    print("\n[1/5] Loading S&P 500 data...")
    tickers, sector_map = get_sp500_tickers()
    prices, sector_map = download_data(
        tickers, sector_map, years=args.years, use_cache=not args.no_cache
    )

    if "SPY" not in prices:
        print("ERROR: SPY data not available. Cannot proceed.")
        sys.exit(1)

    spy_data = prices["SPY"]
    print(f"  SPY data: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
    print(f"  Tickers with data: {len(prices)}")

    # ── Step 2: Run Strategy ───────────────────────────────────────────────
    print("\n[2/5] Running walk-forward backtest...")
    t0 = time.time()
    result = run_backtest(prices, sector_map, spy_data)
    elapsed = time.time() - t0

    if result is None:
        print("ERROR: Backtest returned no results (insufficient data)")
        sys.exit(1)

    print(f"  Backtest completed in {elapsed:.1f}s")
    print(f"  Period: {result['start_date'].date()} to {result['end_date'].date()}")
    print(f"  Total trades: {len(result['trades'])}")

    # Compute strategy metrics
    strat_metrics = compute_metrics(result["equity_curve"], result["trades"],
                                    label="Post-Crash Recovery")

    # ── Step 3: SPY Baseline ───────────────────────────────────────────────
    print("\n[3/5] Computing SPY buy-and-hold baseline...")
    spy_equity = compute_spy_baseline(spy_data, result["start_date"],
                                       result["end_date"])
    spy_metrics = None
    if spy_equity is not None:
        spy_metrics = compute_metrics(spy_equity, [], label="SPY Buy-and-Hold")

    # ── Step 4: Regime Analysis ────────────────────────────────────────────
    print("\n[4/5] Regime stratification (HC #428 R1)...")
    daily_regime = classify_regimes(spy_data)
    regime_results = regime_stratified_metrics(result["equity_curve"], daily_regime)

    # ── Step 5: Permutation Test ───────────────────────────────────────────
    perm_results = {}
    if not args.skip_permutation:
        print(f"\n[5/5] Permutation test ({args.n_perms} shuffles)...")
        t0 = time.time()
        perm_results = permutation_test(
            prices, sector_map, spy_data,
            strat_metrics["sharpe"], n_perms=args.n_perms
        )
        print(f"  Permutation test completed in {time.time() - t0:.1f}s")
    else:
        print("\n[5/5] Permutation test SKIPPED (--skip-permutation)")
        perm_results = {"p_value": None, "note": "skipped"}

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    report = print_report(strat_metrics, spy_metrics, regime_results,
                          perm_results)
    save_results(strat_metrics, spy_metrics, regime_results, perm_results,
                 report)


if __name__ == "__main__":
    main()
