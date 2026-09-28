#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) Backtester
===================================================
Detects earnings-like gap events (>3% gap + >2x volume) and trades the drift.

Strategy:
- LONG: stocks gapping UP >3% on earnings with volume >2x avg → ride momentum
- SHORT: stocks gapping DOWN >3% on earnings with volume >2x avg → ride continuation
- Dynamic exits: trailing stop (ATR-based), momentum breakdown, time stop, profit target

Walk-forward: 2-year in-sample, 1-year OOT
Universe: S&P 500 (2018-2026)
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "earnings_drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Strategy parameters
GAP_THRESHOLD = 0.03          # 3% gap to qualify as earnings event
VOLUME_MULTIPLIER = 2.0       # Volume must be 2x 20-day average
MAX_HOLD_DAYS = 60            # Time stop
ATR_PERIOD = 20               # ATR lookback
TRAILING_STOP_ATR = 1.5       # Trailing stop = 1.5x ATR from peak
TIGHT_TRAILING_ATR = 1.0      # Tighten to 1x ATR after profit target
PROFIT_TARGET = 0.15          # 15% gain → tighten trailing stop
MOMENTUM_BREAKDOWN_DAYS = 5   # Check 5-day returns
MOMENTUM_BREAKDOWN_COUNT = 3  # 3 consecutive negative checks → exit
MAX_POSITIONS = 20            # Max concurrent positions
POSITION_SIZE_PCT = 0.05      # 5% of equity per position

# Walk-forward
WF_TRAIN_YEARS = 2
WF_TEST_YEARS = 1
START_YEAR = 2018
END_YEAR = 2026

# ─── S&P 500 Universe ────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Get current S&P 500 constituents. Falls back to cached list."""
    cache_file = CACHE_DIR / "sp500_tickers.json"

    # Try scraping Wikipedia
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = table[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        with open(cache_file, "w") as f:
            json.dump(tickers, f)
        log.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        log.warning(f"Wikipedia fetch failed: {e}")

    # Try cache
    if cache_file.exists():
        with open(cache_file) as f:
            tickers = json.load(f)
        log.info(f"Loaded {len(tickers)} cached S&P 500 tickers")
        return tickers

    # Hardcoded fallback — top 100 by weight (enough for a meaningful backtest)
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "TMO", "ACN", "ABT", "DHR", "CRM", "NKE", "CMCSA", "NEE", "VZ",
        "ADBE", "TXN", "PM", "RTX", "HON", "UNP", "INTC", "BMY", "QCOM",
        "T", "LOW", "AMGN", "UPS", "GS", "BA", "CAT", "SPGI", "BLK",
        "ELV", "DE", "GILD", "SYK", "MDLZ", "ADP", "ADI", "MMC", "CI",
        "LMT", "PLD", "CB", "AMT", "ISRG", "TJX", "REGN", "VRTX", "MO",
        "SLB", "ZTS", "NOW", "PANW", "CME", "PYPL", "AON", "SCHW", "SNPS",
        "CDNS", "ICE", "DUK", "SO", "CL", "EMR", "ITW", "FDX", "PNC",
        "GM", "F", "DAL", "WBA", "DXCM", "NFLX", "AMD", "ORCL", "COP",
        "EOG",
    ]


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers. Caches to parquet."""
    data = {}
    cache_file = CACHE_DIR / f"ohlcv_{start}_{end}.pkl"

    if cache_file.exists():
        log.info(f"Loading cached data from {cache_file.name}")
        data = pd.read_pickle(cache_file)
        if isinstance(data, dict) and len(data) > 50:
            return data

    log.info(f"Downloading data for {len(tickers)} tickers ({start} to {end})...")

    # Download in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        log.info(f"  Batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: {len(batch)} tickers")

        try:
            df = yf.download(batch_str, start=start, end=end, group_by="ticker",
                             auto_adjust=True, threads=True, progress=False)

            if len(batch) == 1:
                # Single ticker — different structure
                ticker = batch[0]
                if not df.empty:
                    data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for ticker in batch:
                    try:
                        tdf = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                        tdf = tdf.dropna(how="all")
                        if len(tdf) > 100:
                            data[ticker] = tdf
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            log.warning(f"  Batch download failed: {e}")

    log.info(f"Downloaded data for {len(data)} tickers")

    # Cache
    try:
        pd.to_pickle(data, cache_file)
    except Exception as e:
        log.warning(f"Cache save failed: {e}")

    return data


# ─── SPY Benchmark ───────────────────────────────────────────────────────────

def download_spy(start: str, end: str) -> pd.DataFrame:
    """Download SPY for benchmark comparison."""
    cache_file = CACHE_DIR / f"spy_{start}_{end}.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    # Flatten multi-level columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]]
    pd.to_pickle(spy, cache_file)
    return spy


# ─── Signal Detection ────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def detect_earnings_events(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Detect earnings-like gap events: >3% overnight gap with >2x average volume.
    Returns DataFrame with event details.
    """
    if len(df) < 50:
        return pd.DataFrame()

    close = df["Close"].values
    open_ = df["Open"].values
    volume = df["Volume"].values

    # Overnight gap: (today open - yesterday close) / yesterday close
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    gap_pct = (open_ - prev_close) / prev_close

    # Volume ratio: today volume / 20-day average volume
    vol_series = pd.Series(volume, index=df.index)
    avg_vol = vol_series.rolling(20).mean().shift(1)  # Shift to avoid look-ahead
    vol_ratio = vol_series / avg_vol

    # ATR for trailing stops
    atr = compute_atr(df, ATR_PERIOD)

    events = []
    for i in range(25, len(df)):
        gap = gap_pct[i]
        vr = vol_ratio.iloc[i]

        if np.isnan(gap) or np.isnan(vr):
            continue

        abs_gap = abs(gap)
        if abs_gap >= GAP_THRESHOLD and vr >= VOLUME_MULTIPLIER:
            events.append({
                "date": df.index[i],
                "ticker": ticker,
                "direction": "LONG" if gap > 0 else "SHORT",
                "gap_pct": gap,
                "volume_ratio": vr,
                "entry_price": open_[i],  # Enter at open on gap day
                "close_price": close[i],
                "atr": atr.iloc[i] if not np.isnan(atr.iloc[i]) else close[i] * 0.02,
                "idx": i,
            })

    return pd.DataFrame(events)


# ─── Position & Trade Management ─────────────────────────────────────────────

class Position:
    """Tracks an individual position with dynamic exits."""

    def __init__(self, ticker: str, direction: str, entry_date, entry_price: float,
                 atr: float, shares: int, gap_pct: float):
        self.ticker = ticker
        self.direction = direction  # LONG or SHORT
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.atr = atr
        self.shares = shares
        self.gap_pct = gap_pct

        # Tracking
        self.peak_price = entry_price if direction == "LONG" else entry_price
        self.trough_price = entry_price if direction == "SHORT" else entry_price
        self.days_held = 0
        self.trailing_atr_mult = TRAILING_STOP_ATR
        self.profit_target_hit = False
        self.momentum_neg_count = 0
        self.recent_returns = []
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None

    def update(self, date, high: float, low: float, close: float,
               five_day_return: float) -> bool:
        """
        Update position with new day's data. Returns True if position should be closed.
        """
        self.days_held += 1

        if self.direction == "LONG":
            self.peak_price = max(self.peak_price, high)
            current_return = (close - self.entry_price) / self.entry_price

            # Profit target → tighten trailing stop
            if current_return >= PROFIT_TARGET and not self.profit_target_hit:
                self.profit_target_hit = True
                self.trailing_atr_mult = TIGHT_TRAILING_ATR

            # Trailing stop
            trail_stop = self.peak_price - self.trailing_atr_mult * self.atr
            if low <= trail_stop:
                self.exit_price = max(trail_stop, low)  # Approximate fill
                self.exit_date = date
                self.exit_reason = "trailing_stop"
                return True

        else:  # SHORT
            self.trough_price = min(self.trough_price, low)
            current_return = (self.entry_price - close) / self.entry_price

            if current_return >= PROFIT_TARGET and not self.profit_target_hit:
                self.profit_target_hit = True
                self.trailing_atr_mult = TIGHT_TRAILING_ATR

            # Trailing stop (for shorts, stop is above trough)
            trail_stop = self.trough_price + self.trailing_atr_mult * self.atr
            if high >= trail_stop:
                self.exit_price = min(trail_stop, high)
                self.exit_date = date
                self.exit_reason = "trailing_stop"
                return True

        # Momentum breakdown: 5-day returns negative for 3 consecutive checks
        if not np.isnan(five_day_return):
            if self.direction == "LONG":
                is_negative = five_day_return < 0
            else:
                is_negative = five_day_return > 0  # For shorts, positive 5d return is bad

            if is_negative:
                self.momentum_neg_count += 1
            else:
                self.momentum_neg_count = 0

            if self.momentum_neg_count >= MOMENTUM_BREAKDOWN_COUNT:
                self.exit_price = close
                self.exit_date = date
                self.exit_reason = "momentum_breakdown"
                return True

        # Time stop
        if self.days_held >= MAX_HOLD_DAYS:
            self.exit_price = close
            self.exit_date = date
            self.exit_reason = "time_stop"
            return True

        return False

    def pnl(self) -> float:
        """Return P&L in dollar terms."""
        if self.exit_price is None:
            return 0.0
        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.exit_price) * self.shares

    def pnl_pct(self) -> float:
        """Return P&L as percentage."""
        if self.exit_price is None:
            return 0.0
        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_date": str(self.entry_date.date()) if hasattr(self.entry_date, 'date') else str(self.entry_date),
            "exit_date": str(self.exit_date.date()) if self.exit_date and hasattr(self.exit_date, 'date') else str(self.exit_date),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "gap_pct": round(self.gap_pct * 100, 2),
            "pnl_pct": round(self.pnl_pct() * 100, 2),
            "days_held": self.days_held,
            "exit_reason": self.exit_reason,
            "profit_target_hit": self.profit_target_hit,
        }


# ─── Backtester ──────────────────────────────────────────────────────────────

class EarningsDriftBacktester:
    """Walk-forward PEAD backtester with dynamic exit management."""

    def __init__(self, data: dict[str, pd.DataFrame], spy_data: pd.DataFrame):
        self.data = data
        self.spy = spy_data
        self.all_trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def detect_all_events(self) -> pd.DataFrame:
        """Detect earnings gap events across all tickers."""
        all_events = []
        for ticker, df in self.data.items():
            events = detect_earnings_events(df, ticker)
            if len(events) > 0:
                all_events.append(events)

        if not all_events:
            return pd.DataFrame()

        events_df = pd.concat(all_events, ignore_index=True)
        events_df = events_df.sort_values("date").reset_index(drop=True)
        log.info(f"Detected {len(events_df)} earnings gap events across {events_df['ticker'].nunique()} tickers")
        log.info(f"  LONG signals: {(events_df['direction'] == 'LONG').sum()}")
        log.info(f"  SHORT signals: {(events_df['direction'] == 'SHORT').sum()}")
        return events_df

    def run_backtest(self, events: pd.DataFrame, start_date, end_date,
                     initial_equity: float = 1_000_000) -> list[dict]:
        """
        Run backtest over a specific period.
        Returns list of completed trade dicts.
        """
        if events.empty:
            return []

        # Filter events to period
        mask = (events["date"] >= pd.Timestamp(start_date)) & (events["date"] <= pd.Timestamp(end_date))
        period_events = events[mask].copy()

        if period_events.empty:
            return []

        # Get all trading dates in period
        all_dates = set()
        for ticker, df in self.data.items():
            dates = df.loc[start_date:end_date].index
            all_dates.update(dates)
        all_dates = sorted(all_dates)

        if not all_dates:
            return []

        equity = initial_equity
        positions: list[Position] = []
        completed_trades = []
        daily_equity = []

        # Group events by date for fast lookup
        events_by_date = {}
        for _, ev in period_events.iterrows():
            d = ev["date"]
            if d not in events_by_date:
                events_by_date[d] = []
            events_by_date[d].append(ev)

        for date in all_dates:
            # Update existing positions
            closed = []
            for pos in positions:
                ticker = pos.ticker
                if ticker not in self.data:
                    continue

                df = self.data[ticker]
                if date not in df.index:
                    continue

                row = df.loc[date]

                # Compute 5-day return
                loc = df.index.get_loc(date)
                if loc >= 5:
                    five_day_return = (df["Close"].iloc[loc] - df["Close"].iloc[loc - 5]) / df["Close"].iloc[loc - 5]
                else:
                    five_day_return = np.nan

                should_close = pos.update(date, row["High"], row["Low"], row["Close"], five_day_return)
                if should_close:
                    closed.append(pos)
                    equity += pos.pnl()
                    completed_trades.append(pos.to_dict())

            for pos in closed:
                positions.remove(pos)

            # Open new positions from today's events
            if date in events_by_date and len(positions) < MAX_POSITIONS:
                for ev in events_by_date[date]:
                    if len(positions) >= MAX_POSITIONS:
                        break

                    # Check we don't already have a position in this ticker
                    if any(p.ticker == ev["ticker"] for p in positions):
                        continue

                    # Position sizing: fixed % of current equity
                    position_value = equity * POSITION_SIZE_PCT
                    entry_price = ev["entry_price"]
                    if entry_price <= 0 or np.isnan(entry_price):
                        continue
                    shares = int(position_value / entry_price)
                    if shares <= 0:
                        continue

                    pos = Position(
                        ticker=ev["ticker"],
                        direction=ev["direction"],
                        entry_date=ev["date"],
                        entry_price=entry_price,
                        atr=ev["atr"],
                        shares=shares,
                        gap_pct=ev["gap_pct"],
                    )
                    positions.append(pos)

            # Track daily equity (mark open positions to market)
            unrealized = 0
            for pos in positions:
                if pos.ticker in self.data and date in self.data[pos.ticker].index:
                    current_price = self.data[pos.ticker].loc[date, "Close"]
                    if pos.direction == "LONG":
                        unrealized += (current_price - pos.entry_price) * pos.shares
                    else:
                        unrealized += (pos.entry_price - current_price) * pos.shares

            daily_equity.append({
                "date": date,
                "equity": equity + unrealized,
                "realized_equity": equity,
                "open_positions": len(positions),
            })

        # Force-close any remaining positions at period end
        for pos in positions:
            if pos.ticker in self.data:
                df = self.data[pos.ticker]
                last_date = df.loc[:end_date].index[-1] if len(df.loc[:end_date]) > 0 else None
                if last_date is not None:
                    pos.exit_price = df.loc[last_date, "Close"]
                    pos.exit_date = last_date
                    pos.exit_reason = "period_end"
                    equity += pos.pnl()
                    completed_trades.append(pos.to_dict())

        self.equity_curve.extend(daily_equity)
        return completed_trades

    def run_walk_forward(self) -> dict:
        """
        Walk-forward backtest: 2-year train (parameter selection), 1-year OOT.
        Since we use fixed parameters here, train period is for event detection calibration,
        and OOT is the actual backtest period.
        """
        log.info("=" * 70)
        log.info("WALK-FORWARD PEAD BACKTEST")
        log.info("=" * 70)

        # Detect all events first
        all_events = self.detect_all_events()
        if all_events.empty:
            log.error("No events detected!")
            return {}

        # Define walk-forward windows
        windows = []
        for test_start_year in range(START_YEAR + WF_TRAIN_YEARS, END_YEAR + 1, WF_TEST_YEARS):
            train_start = f"{test_start_year - WF_TRAIN_YEARS}-01-01"
            train_end = f"{test_start_year - 1}-12-31"
            test_start = f"{test_start_year}-01-01"
            test_end = f"{min(test_start_year + WF_TEST_YEARS - 1, END_YEAR)}-12-31"
            windows.append({
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
                "label": f"{test_start_year}",
            })

        log.info(f"\nWalk-forward windows: {len(windows)}")
        for w in windows:
            log.info(f"  Train: {w['train_start']} to {w['train_end']} | Test: {w['test_start']} to {w['test_end']}")

        # Run each OOT window
        oot_trades = []
        is_trades = []

        for w in windows:
            log.info(f"\n--- Window {w['label']}: OOT {w['test_start']} to {w['test_end']} ---")

            # In-sample (for stats only, not used for parameter tuning in this version)
            is_t = self.run_backtest(all_events, w["train_start"], w["train_end"])
            is_trades.extend(is_t)
            log.info(f"  IS trades: {len(is_t)}")

            # Out-of-sample
            oot_t = self.run_backtest(all_events, w["test_start"], w["test_end"])
            oot_trades.extend(oot_t)
            log.info(f"  OOT trades: {len(oot_t)}")

        self.all_trades = oot_trades

        # Compute metrics
        results = self.compute_metrics(oot_trades, "OOT")
        is_results = self.compute_metrics(is_trades, "IS")

        # SPY benchmark
        spy_metrics = self.compute_spy_benchmark()

        # Regime analysis
        regime = self.regime_analysis(oot_trades)

        # Long vs Short breakdown
        long_trades = [t for t in oot_trades if t["direction"] == "LONG"]
        short_trades = [t for t in oot_trades if t["direction"] == "SHORT"]
        long_metrics = self.compute_metrics(long_trades, "LONG")
        short_metrics = self.compute_metrics(short_trades, "SHORT")

        # Exit reason breakdown
        exit_reasons = {}
        for t in oot_trades:
            r = t.get("exit_reason", "unknown")
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "avg_pnl_pct": []}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["avg_pnl_pct"].append(t["pnl_pct"])
        for r in exit_reasons:
            vals = exit_reasons[r]["avg_pnl_pct"]
            exit_reasons[r]["avg_pnl_pct"] = round(np.mean(vals), 2) if vals else 0
            exit_reasons[r]["win_rate"] = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1) if vals else 0

        summary = {
            "strategy": "Post-Earnings Announcement Drift (PEAD)",
            "universe": "S&P 500",
            "period": f"{START_YEAR + WF_TRAIN_YEARS}-{END_YEAR} (OOT)",
            "parameters": {
                "gap_threshold": GAP_THRESHOLD,
                "volume_multiplier": VOLUME_MULTIPLIER,
                "max_hold_days": MAX_HOLD_DAYS,
                "trailing_stop_atr": TRAILING_STOP_ATR,
                "tight_trailing_atr": TIGHT_TRAILING_ATR,
                "profit_target": PROFIT_TARGET,
                "max_positions": MAX_POSITIONS,
                "position_size_pct": POSITION_SIZE_PCT,
            },
            "oot_results": results,
            "is_results": is_results,
            "long_side": long_metrics,
            "short_side": short_metrics,
            "spy_benchmark": spy_metrics,
            "regime_analysis": regime,
            "exit_reason_breakdown": exit_reasons,
            "walk_forward_windows": [w["label"] for w in windows],
        }

        return summary

    def compute_metrics(self, trades: list[dict], label: str) -> dict:
        """Compute strategy metrics from trade list."""
        if not trades:
            return {"error": "no trades", "label": label}

        pnls = np.array([t["pnl_pct"] for t in trades])
        n = len(pnls)

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / n * 100 if n > 0 else 0
        avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
        avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0

        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Approximate daily returns from equity curve for Sharpe/Sortino
        if self.equity_curve:
            eq = pd.DataFrame(self.equity_curve)
            eq = eq.drop_duplicates("date").set_index("date").sort_index()
            daily_rets = eq["equity"].pct_change().dropna()

            if len(daily_rets) > 20:
                sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std() if daily_rets.std() > 0 else 0
                downside = daily_rets[daily_rets < 0]
                sortino = np.sqrt(252) * daily_rets.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

                # Max drawdown
                cum = (1 + daily_rets).cumprod()
                rolling_max = cum.cummax()
                drawdown = (cum - rolling_max) / rolling_max
                max_dd = drawdown.min() * 100

                # CAGR
                total_days = (eq.index[-1] - eq.index[0]).days
                total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0]
                cagr = (total_return ** (365.25 / total_days) - 1) * 100 if total_days > 0 else 0
            else:
                sharpe = sortino = cagr = max_dd = 0
        else:
            sharpe = sortino = cagr = max_dd = 0

        # Trades per year
        if trades:
            dates = [t["entry_date"] for t in trades]
            min_date = min(dates)
            max_date = max(dates)
            years = max((pd.Timestamp(max_date) - pd.Timestamp(min_date)).days / 365.25, 1)
            trades_per_year = n / years
        else:
            trades_per_year = 0

        avg_hold = np.mean([t["days_held"] for t in trades])

        metrics = {
            "label": label,
            "total_trades": n,
            "trades_per_year": round(trades_per_year, 1),
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_hold_days": round(avg_hold, 1),
            "total_return_pct": round(np.sum(pnls) * 100 / MAX_POSITIONS, 2),  # Approximate
            "median_pnl_pct": round(np.median(pnls) * 100, 2),
            "best_trade_pct": round(np.max(pnls) * 100, 2),
            "worst_trade_pct": round(np.min(pnls) * 100, 2),
        }

        return metrics

    def compute_spy_benchmark(self) -> dict:
        """Buy-and-hold SPY over the OOT period."""
        if self.spy is None or self.spy.empty:
            return {"error": "no SPY data"}

        test_start = f"{START_YEAR + WF_TRAIN_YEARS}-01-01"
        test_end = f"{END_YEAR}-12-31"

        spy = self.spy.loc[test_start:test_end]
        if spy.empty:
            return {"error": "no SPY data in period"}

        spy_close = spy["Close"].squeeze()  # Ensure Series not DataFrame
        daily_rets = spy_close.pct_change().dropna()
        total_days = (spy.index[-1] - spy.index[0]).days
        total_return = float(spy_close.iloc[-1] / spy_close.iloc[0])

        cagr = (total_return ** (365.25 / total_days) - 1) * 100 if total_days > 0 else 0
        std = float(daily_rets.std())
        mean = float(daily_rets.mean())
        sharpe = np.sqrt(252) * mean / std if std > 0 else 0
        downside = daily_rets[daily_rets < 0]
        ds_std = float(downside.std()) if len(downside) > 0 else 0
        sortino = np.sqrt(252) * mean / ds_std if ds_std > 0 else 0

        cum = (1 + daily_rets).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100

        return {
            "total_return_pct": round((total_return - 1) * 100, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }

    def regime_analysis(self, trades: list[dict]) -> dict:
        """Classify performance by market regime (bull/bear/flat)."""
        if not trades or self.spy is None or self.spy.empty:
            return {"error": "insufficient data"}

        # Classify each month as bull (>2%), bear (<-2%), or flat
        spy_monthly = self.spy["Close"].resample("ME").last()
        spy_monthly_ret = spy_monthly.pct_change()

        regime_map = {}
        for date, ret in spy_monthly_ret.items():
            if np.isnan(ret):
                continue
            month_key = date.strftime("%Y-%m")
            if ret > 0.02:
                regime_map[month_key] = "bull"
            elif ret < -0.02:
                regime_map[month_key] = "bear"
            else:
                regime_map[month_key] = "flat"

        # Classify trades by regime at entry
        regime_trades = {"bull": [], "bear": [], "flat": []}
        for t in trades:
            entry = pd.Timestamp(t["entry_date"])
            month_key = entry.strftime("%Y-%m")
            regime = regime_map.get(month_key, "flat")
            regime_trades[regime].append(t["pnl_pct"])

        result = {}
        for regime, pnls in regime_trades.items():
            if pnls:
                pnls = np.array(pnls)
                wins = pnls[pnls > 0]
                result[regime] = {
                    "n_trades": len(pnls),
                    "win_rate_pct": round(len(wins) / len(pnls) * 100, 1),
                    "avg_pnl_pct": round(np.mean(pnls) * 100, 2),
                    "sharpe_approx": round(np.mean(pnls) / np.std(pnls), 2) if np.std(pnls) > 0 else 0,
                }
            else:
                result[regime] = {"n_trades": 0}

        # Regime skew check (HC #428)
        sharpes = {r: result[r].get("sharpe_approx", 0) for r in ["bull", "bear", "flat"] if result[r]["n_trades"] > 5}
        if len(sharpes) >= 2:
            max_s = max(abs(v) for v in sharpes.values())
            if max_s > 0:
                values = list(sharpes.values())
                skew = abs(max(values) - min(values)) / max_s
                result["regime_skew"] = round(skew, 2)
                result["regime_skew_pass"] = skew <= 0.50

        return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_results(summary: dict):
    """Pretty-print results."""
    print("\n" + "=" * 70)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) BACKTEST RESULTS")
    print("=" * 70)

    oot = summary.get("oot_results", {})
    spy = summary.get("spy_benchmark", {})

    print(f"\nPeriod: {summary.get('period', 'N/A')}")
    print(f"Universe: {summary.get('universe', 'N/A')}")

    print("\n--- OOT Performance ---")
    print(f"  Total Trades:      {oot.get('total_trades', 0)}")
    print(f"  Trades/Year:       {oot.get('trades_per_year', 0)}")
    print(f"  Win Rate:          {oot.get('win_rate_pct', 0):.1f}%")
    print(f"  Avg Win:           {oot.get('avg_win_pct', 0):+.2f}%")
    print(f"  Avg Loss:          {oot.get('avg_loss_pct', 0):+.2f}%")
    print(f"  Profit Factor:     {oot.get('profit_factor', 0):.2f}")
    print(f"  Sharpe:            {oot.get('sharpe', 0):.2f}")
    print(f"  Sortino:           {oot.get('sortino', 0):.2f}")
    print(f"  CAGR:              {oot.get('cagr_pct', 0):.2f}%")
    print(f"  Max Drawdown:      {oot.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Avg Hold Days:     {oot.get('avg_hold_days', 0):.1f}")
    print(f"  Median Trade:      {oot.get('median_pnl_pct', 0):+.2f}%")
    print(f"  Best Trade:        {oot.get('best_trade_pct', 0):+.2f}%")
    print(f"  Worst Trade:       {oot.get('worst_trade_pct', 0):+.2f}%")

    print("\n--- SPY Buy & Hold Benchmark ---")
    print(f"  Total Return:      {spy.get('total_return_pct', 0):.2f}%")
    print(f"  CAGR:              {spy.get('cagr_pct', 0):.2f}%")
    print(f"  Sharpe:            {spy.get('sharpe', 0):.2f}")
    print(f"  Sortino:           {spy.get('sortino', 0):.2f}")
    print(f"  Max Drawdown:      {spy.get('max_drawdown_pct', 0):.2f}%")

    # Long vs Short
    long_m = summary.get("long_side", {})
    short_m = summary.get("short_side", {})
    print("\n--- Long vs Short Breakdown ---")
    print(f"  {'Metric':<20} {'LONG':>12} {'SHORT':>12}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12}")
    for key in ["total_trades", "win_rate_pct", "avg_win_pct", "avg_loss_pct", "profit_factor", "avg_hold_days"]:
        lv = long_m.get(key, "N/A")
        sv = short_m.get(key, "N/A")
        lv_str = f"{lv}" if isinstance(lv, (int, str)) else f"{lv:.2f}"
        sv_str = f"{sv}" if isinstance(sv, (int, str)) else f"{sv:.2f}"
        print(f"  {key:<20} {lv_str:>12} {sv_str:>12}")

    # Regime
    regime = summary.get("regime_analysis", {})
    print("\n--- Regime Analysis ---")
    for r in ["bull", "bear", "flat"]:
        rd = regime.get(r, {})
        if rd.get("n_trades", 0) > 0:
            print(f"  {r.upper():6}: {rd['n_trades']:3d} trades | WR {rd['win_rate_pct']:.1f}% | Avg {rd['avg_pnl_pct']:+.2f}% | Sharpe(approx) {rd.get('sharpe_approx', 0):.2f}")

    skew_pass = regime.get("regime_skew_pass")
    if skew_pass is not None:
        print(f"  Regime Skew:       {regime.get('regime_skew', 'N/A')} ({'PASS' if skew_pass else 'FAIL'} — threshold 0.50)")

    # Exit reasons
    exits = summary.get("exit_reason_breakdown", {})
    if exits:
        print("\n--- Exit Reason Breakdown ---")
        for reason, info in sorted(exits.items(), key=lambda x: -x[1]["count"]):
            print(f"  {reason:<25} {info['count']:4d} trades | Avg P&L {info['avg_pnl_pct']:+.2f}% | WR {info.get('win_rate', 0):.1f}%")

    print("\n" + "=" * 70)


def main():
    log.info("Starting PEAD Backtester...")

    # Get tickers
    tickers = get_sp500_tickers()

    # Download data
    start_date = f"{START_YEAR}-01-01"
    end_date = f"{END_YEAR}-12-31"
    data = download_data(tickers, start_date, end_date)
    spy = download_spy(start_date, end_date)

    if not data:
        log.error("No data downloaded. Exiting.")
        sys.exit(1)

    # Run backtest
    bt = EarningsDriftBacktester(data, spy)
    summary = bt.run_walk_forward()

    if not summary:
        log.error("Backtest produced no results.")
        sys.exit(1)

    # Print results
    print_results(summary)

    # Save results
    summary_file = OUTPUT_DIR / "backtest_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nSummary saved to {summary_file}")

    # Save trades
    trades_file = OUTPUT_DIR / "all_trades.json"
    with open(trades_file, "w") as f:
        json.dump(bt.all_trades, f, indent=2, default=str)
    log.info(f"Trades saved to {trades_file}")

    # Save equity curve
    if bt.equity_curve:
        eq_df = pd.DataFrame(bt.equity_curve)
        eq_df.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
        log.info(f"Equity curve saved to {OUTPUT_DIR / 'equity_curve.csv'}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
