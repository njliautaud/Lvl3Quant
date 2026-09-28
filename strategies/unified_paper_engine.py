#!/usr/bin/env python3
"""
Unified Paper Trading Engine
Runs ALL validated strategies side-by-side, tracks performance.
Designed to run daily via cron.

Usage:
    python3 unified_paper_engine.py           # Normal daily run
    python3 unified_paper_engine.py --reset    # Reset all state
    python3 unified_paper_engine.py --backfill 60  # Backfill N days
"""

import json
import csv
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

STATE_FILE = "/home/jupiter/Lvl3Quant/data/paper_positions.json"
HISTORY_FILE = "/home/jupiter/Lvl3Quant/data/paper_trade_history.csv"

QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO",
    "JPM", "UNH", "LLY", "V", "MA", "ABBV", "COST", "HD",
    "PG", "JNJ", "MRK", "PEP", "KO", "WMT",
]

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU",
]

EXTRA_TICKERS = ["^TNX", "^VIX", "SPY"]

ALL_TICKERS = list(set(QUALITY_UNIVERSE + SECTOR_ETFS + EXTRA_TICKERS))

HISTORY_FIELDS = [
    "date", "strategy", "action", "ticker", "shares", "price",
    "position_value", "realized_pnl", "notes",
]


# ─── DATA LAYER ───────────────────────────────────────────────────────────────

class MarketData:
    """Downloads and caches market data for the session."""

    def __init__(self, lookback_days: int = 120):
        self.lookback_days = lookback_days
        self.data: Dict[str, pd.DataFrame] = {}
        self._downloaded = False

    def download(self, as_of_date: Optional[str] = None):
        """Download all required data. as_of_date for backfill mode."""
        end = pd.Timestamp(as_of_date) + timedelta(days=1) if as_of_date else pd.Timestamp.now()
        start = end - timedelta(days=self.lookback_days + 10)

        print(f"Downloading market data for {len(ALL_TICKERS)} tickers...")
        try:
            raw = yf.download(
                ALL_TICKERS, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"), progress=False, group_by="ticker",
                auto_adjust=True, threads=True,
            )
        except Exception as e:
            print(f"  WARNING: Batch download failed ({e}), trying individually...")
            raw = None

        for ticker in ALL_TICKERS:
            try:
                if raw is not None and len(ALL_TICKERS) > 1:
                    if ticker in raw.columns.get_level_values(0):
                        df = raw[ticker].dropna(how="all")
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

                if df.empty or len(df) < 5:
                    df = yf.download(
                        ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True,
                    )

                if not df.empty:
                    # Flatten multi-level columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=["Close"])
                    self.data[ticker] = df
            except Exception as e:
                print(f"  WARNING: Could not download {ticker}: {e}")

        self._downloaded = True
        loaded = len(self.data)
        print(f"  Loaded data for {loaded}/{len(ALL_TICKERS)} tickers")
        if loaded == 0:
            print("  CRITICAL: No data loaded. Check internet connection.")
            sys.exit(1)

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        return self.data.get(ticker)

    def latest_price(self, ticker: str) -> Optional[float]:
        df = self.get(ticker)
        if df is not None and len(df) > 0:
            return float(df["Close"].iloc[-1])
        return None

    def latest_open(self, ticker: str) -> Optional[float]:
        df = self.get(ticker)
        if df is not None and len(df) > 0:
            return float(df["Open"].iloc[-1])
        return None

    def latest_date(self) -> str:
        """Return the latest trading date across all data."""
        latest = None
        for df in self.data.values():
            if len(df) > 0:
                dt = df.index[-1]
                if latest is None or dt > latest:
                    latest = dt
        return latest.strftime("%Y-%m-%d") if latest else datetime.now().strftime("%Y-%m-%d")


# ─── INDICATOR HELPERS ────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_vol(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility from daily returns."""
    returns = series.pct_change()
    return returns.rolling(period).std() * np.sqrt(252) * 100


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def consecutive_red_days(df: pd.DataFrame) -> int:
    """Count consecutive red (close < open) days ending at the last row."""
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df["Close"].iloc[i] < df["Open"].iloc[i]:
            count += 1
        else:
            break
    return count


def is_green_day(df: pd.DataFrame, idx: int = -1) -> bool:
    return df["Close"].iloc[idx] >= df["Open"].iloc[idx]


# ─── STATE MANAGEMENT ─────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return init_state()


def init_state() -> dict:
    return {
        "created": datetime.now().isoformat(),
        "last_run": None,
        "positions": {},       # strategy_name -> [position dicts]
        "closed_trades": {},   # strategy_name -> [closed trade dicts]
        "portfolio_state": {}, # for rebalancing strategies
        "signals_today": [],
        "version": 2,
    }


def save_state(state: dict):
    state["last_run"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_trade_history(rows: List[dict]):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})


# ─── STRATEGY BASE ────────────────────────────────────────────────────────────

class Strategy:
    """Base class for all strategies."""

    name: str = "base"
    group: str = "A"  # A=agentic, B=portfolio
    max_concurrent: int = 2
    hold_days: int = 10
    trade_size: float = 300.0
    options_candidate: bool = False

    def __init__(self, market: MarketData, state: dict):
        self.market = market
        self.state = state
        if self.name not in self.state["positions"]:
            self.state["positions"][self.name] = []
        if self.name not in self.state["closed_trades"]:
            self.state["closed_trades"][self.name] = []

    @property
    def positions(self) -> list:
        return self.state["positions"][self.name]

    @positions.setter
    def positions(self, val):
        self.state["positions"][self.name] = val

    @property
    def closed_trades(self) -> list:
        return self.state["closed_trades"][self.name]

    def active_count(self) -> int:
        return len(self.positions)

    def scan_signals(self, today: str) -> List[dict]:
        """Return list of signal dicts: {ticker, reason, price, ...}"""
        raise NotImplementedError

    def check_exits(self, today: str) -> List[dict]:
        """Check if any positions should be closed. Return trade records."""
        exits = []
        remaining = []
        for pos in self.positions:
            entry_date = pd.Timestamp(pos["entry_date"])
            current_date = pd.Timestamp(today)
            days_held = np.busday_count(
                entry_date.date(), current_date.date()
            )
            if days_held >= self.hold_days:
                exit_price = self.market.latest_price(pos["ticker"])
                if exit_price is None:
                    remaining.append(pos)
                    continue
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                trade = {
                    "date": today,
                    "strategy": self.name,
                    "action": "SELL",
                    "ticker": pos["ticker"],
                    "shares": pos["shares"],
                    "price": exit_price,
                    "position_value": exit_price * pos["shares"],
                    "realized_pnl": round(pnl, 2),
                    "notes": f"Hold {days_held}d. Entry={pos['entry_price']:.2f}",
                }
                exits.append(trade)
                self.closed_trades.append({
                    **pos,
                    "exit_date": today,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "days_held": days_held,
                })
            else:
                remaining.append(pos)
        self.positions = remaining
        return exits

    def execute_entries(self, signals: List[dict], today: str) -> List[dict]:
        """Execute entry trades from signals. Return trade records."""
        entries = []
        for sig in signals:
            if self.active_count() >= self.max_concurrent:
                break
            # Check no duplicate ticker in active positions
            if any(p["ticker"] == sig["ticker"] for p in self.positions):
                continue
            price = sig.get("price") or self.market.latest_open(sig["ticker"])
            if price is None or price <= 0:
                continue
            shares = max(1, int(self.trade_size / price))
            pos = {
                "ticker": sig["ticker"],
                "entry_date": today,
                "entry_price": round(price, 2),
                "shares": shares,
                "reason": sig.get("reason", ""),
            }
            self.positions.append(pos)
            trade = {
                "date": today,
                "strategy": self.name,
                "action": "BUY",
                "ticker": sig["ticker"],
                "shares": shares,
                "price": round(price, 2),
                "position_value": round(price * shares, 2),
                "realized_pnl": 0,
                "notes": sig.get("reason", ""),
            }
            entries.append(trade)
        return entries

    def unrealized_pnl(self) -> float:
        total = 0.0
        for pos in self.positions:
            current = self.market.latest_price(pos["ticker"])
            if current:
                total += (current - pos["entry_price"]) * pos["shares"]
        return total

    def realized_pnl(self) -> float:
        return sum(t.get("pnl", 0) for t in self.closed_trades)

    def total_pnl(self) -> float:
        return self.realized_pnl() + self.unrealized_pnl()

    def win_rate(self) -> Optional[float]:
        closed = self.closed_trades
        if len(closed) < 2:
            return None
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        return wins / len(closed)

    def sharpe(self) -> Optional[float]:
        closed = self.closed_trades
        if len(closed) < 5:
            return None
        pnls = [t.get("pnl", 0) for t in closed]
        mean = np.mean(pnls)
        std = np.std(pnls, ddof=1)
        if std == 0:
            return None
        # Annualize assuming ~1 trade per week
        return (mean / std) * np.sqrt(52)


# ─── GROUP A: AGENTIC STRATEGIES ──────────────────────────────────────────────

class MultiTFDualSignalL(Strategy):
    name = "multi_tf_dual_signal_l"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            high_20 = close.rolling(20).max()
            pct_below = (close - high_20) / high_20

            rsi_vals = rsi(close, 14)
            cur_rsi = rsi_vals.iloc[-1]

            # >5% below 20-day high
            if pct_below.iloc[-1] > -0.05:
                continue
            # RSI < 35
            if cur_rsi >= 35:
                continue
            # First green day after 3+ red days
            if not is_green_day(df, -1):
                continue
            red_count = 0
            for i in range(len(df) - 2, max(len(df) - 12, -1), -1):
                if df["Close"].iloc[i] < df["Open"].iloc[i]:
                    red_count += 1
                else:
                    break
            if red_count < 3:
                continue
            # Weekly RSI declining 2+ weeks (use 5-day rolling RSI as proxy)
            if len(rsi_vals) >= 15:
                rsi_w1 = rsi_vals.iloc[-6:-1].mean()
                rsi_w2 = rsi_vals.iloc[-11:-6].mean()
                if rsi_w1 >= rsi_w2:
                    continue
            else:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"MultiTF: {pct_below.iloc[-1]*100:.1f}% below 20d high, RSI={cur_rsi:.0f}, {red_count} red days",
            })
        return signals


class RSIDivergenceC(Strategy):
    name = "rsi_divergence_c"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            rsi_vals = rsi(close, 14)

            # New 20-day price low
            if close.iloc[-1] > close.iloc[-20:].min() * 1.005:
                continue

            # Find previous 20-day low (look back 5-15 days ago)
            prev_window = close.iloc[-20:-5]
            if len(prev_window) < 5:
                continue
            prev_low_idx = prev_window.idxmin()
            prev_low_pos = df.index.get_loc(prev_low_idx)
            cur_pos = len(df) - 1

            # RSI makes higher low (bullish divergence)
            if rsi_vals.iloc[cur_pos] <= rsi_vals.iloc[prev_low_pos]:
                continue

            # Volume declining
            if "Volume" in df.columns and len(df) >= 10:
                vol_recent = df["Volume"].iloc[-5:].mean()
                vol_prior = df["Volume"].iloc[-10:-5].mean()
                if vol_prior > 0 and vol_recent >= vol_prior:
                    continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"RSI Div: price new low but RSI higher low ({rsi_vals.iloc[-1]:.0f} vs {rsi_vals.iloc[prev_low_pos]:.0f})",
            })
        return signals


class DualSignalD(Strategy):
    name = "dual_signal_d"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            high_20 = close.rolling(20).max()
            pct_below = (close - high_20) / high_20
            rsi_vals = rsi(close, 14)

            if pct_below.iloc[-1] > -0.05:
                continue
            if rsi_vals.iloc[-1] >= 35:
                continue
            if not is_green_day(df, -1):
                continue
            red_count = 0
            for i in range(len(df) - 2, max(len(df) - 12, -1), -1):
                if df["Close"].iloc[i] < df["Open"].iloc[i]:
                    red_count += 1
                else:
                    break
            if red_count < 3:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"DualD: {pct_below.iloc[-1]*100:.1f}% below high, RSI={rsi_vals.iloc[-1]:.0f}, {red_count} red days",
            })
        return signals


class BondYieldSignal(Strategy):
    name = "bond_yield_signal"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        tnx = self.market.get("^TNX")
        if tnx is None or len(tnx) < 10:
            return signals
        tnx = tnx.loc[:today]
        if len(tnx) < 6:
            return signals

        # 10Y yield drop > 0.1% in 5 trading days
        yield_now = tnx["Close"].iloc[-1]
        yield_5d = tnx["Close"].iloc[-6] if len(tnx) >= 6 else yield_now
        yield_drop = yield_5d - yield_now  # positive = yields falling

        if yield_drop < 0.1:
            return signals

        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            sma_20 = sma(close, 20)
            pct_below_sma = (close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]

            if pct_below_sma > -0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"BondYield: 10Y dropped {yield_drop:.2f}%, {ticker} {pct_below_sma*100:.1f}% below 20-SMA",
            })
        return signals


class IVRVGapEntry(Strategy):
    name = "iv_rv_gap_entry"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        vix_df = self.market.get("^VIX")
        spy_df = self.market.get("SPY")
        if vix_df is None or spy_df is None or len(spy_df) < 25:
            return signals
        vix_df = vix_df.loc[:today]
        spy_df = spy_df.loc[:today]
        if len(vix_df) < 1 or len(spy_df) < 25:
            return signals

        vix_now = vix_df["Close"].iloc[-1]
        spy_rv = realized_vol(spy_df["Close"], 20)
        if len(spy_rv.dropna()) < 1:
            return signals
        rv_now = spy_rv.iloc[-1]

        # VIX exceeds SPY 20-day RV by 5+ points
        if vix_now - rv_now < 5:
            return signals

        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            high_20 = close.rolling(20).max()
            pct_below = (close.iloc[-1] - high_20.iloc[-1]) / high_20.iloc[-1]

            if pct_below > -0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"IV-RV: VIX={vix_now:.1f} vs RV={rv_now:.1f} (gap={vix_now-rv_now:.1f}), {pct_below*100:.1f}% below 20d high",
            })
        return signals


class LiquiditySignal(Strategy):
    name = "liquidity_signal"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 65:
                continue
            df = df.loc[:today]
            if len(df) < 65:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]

            # H-L spread relative to close
            hl_spread = (high - low) / close
            hl_avg_60 = hl_spread.rolling(60).mean()

            # Current spread below 60-day average
            if hl_spread.iloc[-1] >= hl_avg_60.iloc[-1]:
                continue

            # >5% below 20-day high
            high_20 = close.rolling(20).max()
            pct_below = (close.iloc[-1] - high_20.iloc[-1]) / high_20.iloc[-1]
            if pct_below > -0.05:
                continue

            rsi_vals = rsi(close, 14)
            if rsi_vals.iloc[-1] >= 40:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"Liquidity: spread={hl_spread.iloc[-1]*100:.2f}% vs avg={hl_avg_60.iloc[-1]*100:.2f}%, RSI={rsi_vals.iloc[-1]:.0f}",
            })
        return signals


class ConsecutiveDip(Strategy):
    name = "consecutive_dip"
    group = "A"
    hold_days = 10
    trade_size = 300.0
    max_concurrent = 2
    options_candidate = True

    def scan_signals(self, today: str) -> List[dict]:
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            opens = df["Open"]

            # Check last 3+ days are red with increasing losses
            daily_returns = (close / opens - 1)
            losses = []
            for i in range(len(df) - 1, max(len(df) - 8, -1), -1):
                ret = daily_returns.iloc[i]
                if ret < 0:
                    losses.append(abs(ret))
                else:
                    break

            if len(losses) < 3:
                continue

            # Each day's loss bigger than previous (losses list is reverse chronological)
            increasing = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
            if not increasing:
                continue

            # >5% below 20-day high
            high_20 = close.rolling(20).max()
            pct_below = (close.iloc[-1] - high_20.iloc[-1]) / high_20.iloc[-1]
            if pct_below > -0.05:
                continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"ConsecDip: {len(losses)} red days with increasing losses, {pct_below*100:.1f}% below high",
            })
        return signals


class EarningsSurprisePEAD(Strategy):
    name = "earnings_surprise_pead"
    group = "A"
    hold_days = 5
    trade_size = 200.0
    max_concurrent = 2

    def scan_signals(self, today: str) -> List[dict]:
        """
        Buy if stock gapped up >5% on what looks like an earnings day.
        Proxy: gap up >5% with volume >2x 20-day average (earnings-like event).
        """
        signals = []
        for ticker in QUALITY_UNIVERSE:
            df = self.market.get(ticker)
            if df is None or len(df) < 25:
                continue
            df = df.loc[:today]
            if len(df) < 25:
                continue

            close = df["Close"]
            opens = df["Open"]

            # Gap up >5% (today's open vs yesterday's close)
            if len(close) < 2:
                continue
            gap = (opens.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            if gap < 0.05:
                continue

            # Volume confirmation: >2x average
            if "Volume" in df.columns:
                vol_avg = df["Volume"].iloc[-21:-1].mean()
                if vol_avg > 0 and df["Volume"].iloc[-1] < 2 * vol_avg:
                    continue

            signals.append({
                "ticker": ticker,
                "price": float(close.iloc[-1]),
                "reason": f"PEAD: {gap*100:.1f}% gap up with high volume",
            })
        return signals


# ─── GROUP B: PORTFOLIO MANAGEMENT ────────────────────────────────────────────

class SectorETFRotation(Strategy):
    name = "sector_etf_rotation"
    group = "B"
    hold_days = 21  # Monthly
    trade_size = 1000.0
    max_concurrent = 3

    def scan_signals(self, today: str) -> List[dict]:
        """Monthly rebalance: buy top 3 sectors by 1-month return."""
        # Check if we need to rebalance (monthly)
        if self.positions:
            entry = pd.Timestamp(self.positions[0]["entry_date"])
            if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
                return []

        returns = {}
        for etf in SECTOR_ETFS:
            df = self.market.get(etf)
            if df is None or len(df) < 22:
                continue
            df = df.loc[:today]
            if len(df) < 22:
                continue
            ret = (df["Close"].iloc[-1] / df["Close"].iloc[-22]) - 1
            returns[etf] = ret

        if len(returns) < 3:
            return []

        top3 = sorted(returns.items(), key=lambda x: x[1], reverse=True)[:3]
        signals = []
        for etf, ret in top3:
            if any(p["ticker"] == etf for p in self.positions):
                continue
            price = self.market.latest_price(etf)
            if price:
                signals.append({
                    "ticker": etf,
                    "price": price,
                    "reason": f"SectorRot: top 3, 1M return={ret*100:.1f}%",
                })
        return signals

    def check_exits(self, today: str) -> List[dict]:
        """On rebalance, exit all positions first."""
        exits = []
        # Check if rebalance needed
        if not self.positions:
            return exits
        entry = pd.Timestamp(self.positions[0]["entry_date"])
        if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
            return exits

        # Exit all
        remaining = []
        for pos in self.positions:
            exit_price = self.market.latest_price(pos["ticker"])
            if exit_price is None:
                remaining.append(pos)
                continue
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            trade = {
                "date": today,
                "strategy": self.name,
                "action": "SELL",
                "ticker": pos["ticker"],
                "shares": pos["shares"],
                "price": exit_price,
                "position_value": exit_price * pos["shares"],
                "realized_pnl": round(pnl, 2),
                "notes": f"Monthly rebalance. Entry={pos['entry_price']:.2f}",
            }
            exits.append(trade)
            self.closed_trades.append({
                **pos, "exit_date": today, "exit_price": exit_price,
                "pnl": round(pnl, 2), "days_held": 21,
            })
        self.positions = remaining
        return exits


class RiskParity(Strategy):
    name = "risk_parity"
    group = "B"
    hold_days = 21
    trade_size = 10000.0
    max_concurrent = 11

    def scan_signals(self, today: str) -> List[dict]:
        """Monthly rebalance: weight inversely by 60-day realized vol."""
        if self.positions:
            entry = pd.Timestamp(self.positions[0]["entry_date"])
            if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
                return []

        vols = {}
        for etf in SECTOR_ETFS:
            df = self.market.get(etf)
            if df is None or len(df) < 65:
                continue
            df = df.loc[:today]
            if len(df) < 65:
                continue
            rv = realized_vol(df["Close"], 60)
            if len(rv.dropna()) > 0 and rv.iloc[-1] > 0:
                vols[etf] = rv.iloc[-1]

        if len(vols) < 3:
            return []

        # Inverse vol weighting
        inv_vols = {k: 1.0 / v for k, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {k: v / total_inv for k, v in inv_vols.items()}

        signals = []
        for etf, w in weights.items():
            alloc = self.trade_size * w
            price = self.market.latest_price(etf)
            if price and price > 0:
                shares = max(1, int(alloc / price))
                signals.append({
                    "ticker": etf,
                    "price": price,
                    "reason": f"RiskParity: weight={w*100:.1f}%, vol={vols[etf]:.1f}%",
                    "_shares_override": shares,
                })
        return signals

    def execute_entries(self, signals: List[dict], today: str) -> List[dict]:
        entries = []
        for sig in signals:
            if any(p["ticker"] == sig["ticker"] for p in self.positions):
                continue
            price = sig["price"]
            shares = sig.get("_shares_override", max(1, int(self.trade_size / 11 / price)))
            pos = {
                "ticker": sig["ticker"],
                "entry_date": today,
                "entry_price": round(price, 2),
                "shares": shares,
                "reason": sig.get("reason", ""),
            }
            self.positions.append(pos)
            entries.append({
                "date": today,
                "strategy": self.name,
                "action": "BUY",
                "ticker": sig["ticker"],
                "shares": shares,
                "price": round(price, 2),
                "position_value": round(price * shares, 2),
                "realized_pnl": 0,
                "notes": sig.get("reason", ""),
            })
        return entries

    def check_exits(self, today: str) -> List[dict]:
        exits = []
        if not self.positions:
            return exits
        entry = pd.Timestamp(self.positions[0]["entry_date"])
        if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
            return exits
        remaining = []
        for pos in self.positions:
            exit_price = self.market.latest_price(pos["ticker"])
            if exit_price is None:
                remaining.append(pos)
                continue
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            exits.append({
                "date": today, "strategy": self.name, "action": "SELL",
                "ticker": pos["ticker"], "shares": pos["shares"],
                "price": exit_price,
                "position_value": exit_price * pos["shares"],
                "realized_pnl": round(pnl, 2),
                "notes": f"Monthly rebalance. Entry={pos['entry_price']:.2f}",
            })
            self.closed_trades.append({
                **pos, "exit_date": today, "exit_price": exit_price,
                "pnl": round(pnl, 2), "days_held": 21,
            })
        self.positions = remaining
        return exits


class FactorRotation(Strategy):
    name = "factor_rotation"
    group = "B"
    hold_days = 21
    trade_size = 5000.0
    max_concurrent = 3

    def scan_signals(self, today: str) -> List[dict]:
        """Monthly: rotate based on VIX level."""
        if self.positions:
            entry = pd.Timestamp(self.positions[0]["entry_date"])
            if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
                return []

        vix_df = self.market.get("^VIX")
        if vix_df is None or len(vix_df) < 1:
            return []
        vix_df = vix_df.loc[:today]
        vix_now = vix_df["Close"].iloc[-1]

        # Determine allocation
        if vix_now < 15:
            allocs = {"XLK": 1.0}
        elif vix_now <= 25:
            allocs = {"XLK": 0.5, "XLV": 0.5}
        else:
            allocs = {"XLV": 1.0}

        signals = []
        for etf, weight in allocs.items():
            if any(p["ticker"] == etf for p in self.positions):
                continue
            price = self.market.latest_price(etf)
            if price and price > 0:
                alloc_dollars = self.trade_size * weight
                shares = max(1, int(alloc_dollars / price))
                signals.append({
                    "ticker": etf,
                    "price": price,
                    "reason": f"FactorRot: VIX={vix_now:.1f}, {etf} weight={weight*100:.0f}%",
                    "_shares_override": shares,
                })
        return signals

    def execute_entries(self, signals: List[dict], today: str) -> List[dict]:
        entries = []
        for sig in signals:
            if any(p["ticker"] == sig["ticker"] for p in self.positions):
                continue
            price = sig["price"]
            shares = sig.get("_shares_override", max(1, int(self.trade_size / price)))
            pos = {
                "ticker": sig["ticker"],
                "entry_date": today,
                "entry_price": round(price, 2),
                "shares": shares,
                "reason": sig.get("reason", ""),
            }
            self.positions.append(pos)
            entries.append({
                "date": today, "strategy": self.name, "action": "BUY",
                "ticker": sig["ticker"], "shares": shares,
                "price": round(price, 2),
                "position_value": round(price * shares, 2),
                "realized_pnl": 0,
                "notes": sig.get("reason", ""),
            })
        return entries

    def check_exits(self, today: str) -> List[dict]:
        exits = []
        if not self.positions:
            return exits
        entry = pd.Timestamp(self.positions[0]["entry_date"])
        if np.busday_count(entry.date(), pd.Timestamp(today).date()) < 21:
            return exits
        remaining = []
        for pos in self.positions:
            exit_price = self.market.latest_price(pos["ticker"])
            if exit_price is None:
                remaining.append(pos)
                continue
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            exits.append({
                "date": today, "strategy": self.name, "action": "SELL",
                "ticker": pos["ticker"], "shares": pos["shares"],
                "price": exit_price,
                "position_value": exit_price * pos["shares"],
                "realized_pnl": round(pnl, 2),
                "notes": f"Monthly rebalance. Entry={pos['entry_price']:.2f}",
            })
            self.closed_trades.append({
                **pos, "exit_date": today, "exit_price": exit_price,
                "pnl": round(pnl, 2), "days_held": 21,
            })
        self.positions = remaining
        return exits


# ─── ENGINE ───────────────────────────────────────────────────────────────────

ALL_STRATEGIES = [
    MultiTFDualSignalL,
    RSIDivergenceC,
    DualSignalD,
    BondYieldSignal,
    IVRVGapEntry,
    LiquiditySignal,
    ConsecutiveDip,
    EarningsSurprisePEAD,
    SectorETFRotation,
    RiskParity,
    FactorRotation,
]


def run_engine(backfill_date: Optional[str] = None):
    """Run the paper trading engine for today (or a specific date for backfill)."""
    market = MarketData(lookback_days=120)
    market.download(as_of_date=backfill_date)

    today = backfill_date or market.latest_date()
    state = load_state()

    print(f"\n{'='*70}")
    print(f"  UNIFIED PAPER TRADING ENGINE — {today}")
    print(f"{'='*70}")

    strategies = [cls(market, state) for cls in ALL_STRATEGIES]
    all_trades = []
    signals_today = []

    # Phase 1: Check exits
    print(f"\n--- EXITS ---")
    for strat in strategies:
        exits = strat.check_exits(today)
        if exits:
            for t in exits:
                print(f"  EXIT  {strat.name:30s} | {t['ticker']:5s} | PnL: ${t['realized_pnl']:+.2f}")
            all_trades.extend(exits)

    # Phase 2: Scan signals
    print(f"\n--- SIGNALS ---")
    for strat in strategies:
        sigs = strat.scan_signals(today)
        if sigs:
            for s in sigs:
                tag = " [OPTIONS CANDIDATE]" if strat.options_candidate else ""
                print(f"  SIGNAL {strat.name:28s} | {s['ticker']:5s} | {s['reason']}{tag}")
                signals_today.append({
                    "strategy": strat.name,
                    "ticker": s["ticker"],
                    "reason": s["reason"],
                    "options_candidate": strat.options_candidate,
                })

    # Phase 3: Execute entries
    print(f"\n--- ENTRIES ---")
    for strat in strategies:
        sigs = strat.scan_signals(today)
        entries = strat.execute_entries(sigs, today)
        if entries:
            for t in entries:
                print(f"  BUY   {strat.name:30s} | {t['ticker']:5s} | {t['shares']} shares @ ${t['price']:.2f}")
            all_trades.extend(entries)

    if not all_trades and not signals_today:
        print("  (no signals or trades today)")

    # Save trade history
    if all_trades:
        append_trade_history(all_trades)

    state["signals_today"] = signals_today

    # Phase 4: Summary
    print(f"\n{'='*70}")
    print(f"  DAILY SUMMARY — {today}")
    print(f"{'='*70}")

    header = f"{'Strategy':<30s} {'Group':>5s} {'Pos':>4s} {'Realized':>10s} {'Unreal':>10s} {'Total':>10s} {'Trades':>6s} {'WR':>6s} {'Sharpe':>7s}"
    print(f"\n{header}")
    print("-" * len(header))

    total_realized = 0
    total_unrealized = 0
    total_trades_count = 0

    for strat in strategies:
        r_pnl = strat.realized_pnl()
        u_pnl = strat.unrealized_pnl()
        t_pnl = r_pnl + u_pnl
        n_closed = len(strat.closed_trades)
        n_active = strat.active_count()
        wr = strat.win_rate()
        sh = strat.sharpe()

        wr_str = f"{wr*100:.0f}%" if wr is not None else "---"
        sh_str = f"{sh:.2f}" if sh is not None else "---"

        print(f"{strat.name:<30s} {strat.group:>5s} {n_active:>4d} {r_pnl:>+10.2f} {u_pnl:>+10.2f} {t_pnl:>+10.2f} {n_closed:>6d} {wr_str:>6s} {sh_str:>7s}")

        total_realized += r_pnl
        total_unrealized += u_pnl
        total_trades_count += n_closed

    print("-" * len(header))
    total_all = total_realized + total_unrealized
    print(f"{'TOTAL':<30s} {'':>5s} {'':>4s} {total_realized:>+10.2f} {total_unrealized:>+10.2f} {total_all:>+10.2f} {total_trades_count:>6d}")

    # Active positions detail
    any_positions = any(strat.active_count() > 0 for strat in strategies)
    if any_positions:
        print(f"\n--- ACTIVE POSITIONS ---")
        for strat in strategies:
            for pos in strat.positions:
                cur = market.latest_price(pos["ticker"])
                if cur:
                    pnl = (cur - pos["entry_price"]) * pos["shares"]
                    print(f"  {strat.name:28s} | {pos['ticker']:5s} | {pos['shares']} @ ${pos['entry_price']:.2f} -> ${cur:.2f} | PnL: ${pnl:+.2f} | Since: {pos['entry_date']}")

    # Options candidates summary
    opt_candidates = [s for s in signals_today if s.get("options_candidate")]
    if opt_candidates:
        print(f"\n--- OPTIONS CANDIDATES (flagged, not executed) ---")
        for s in opt_candidates:
            print(f"  {s['strategy']:28s} | {s['ticker']:5s} | Consider selling puts")

    save_state(state)
    print(f"\nState saved to {STATE_FILE}")
    print(f"Trade history: {HISTORY_FILE}")
    return state


def backfill(days: int):
    """Run the engine over the last N trading days to build up trade history."""
    print(f"Backfilling {days} trading days...")

    # Get trading dates
    spy = yf.download("SPY", period=f"{days + 10}d", progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    trading_dates = [d.strftime("%Y-%m-%d") for d in spy.index[-days:]]

    # Reset state for clean backfill
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

    for date in trading_dates:
        print(f"\n{'#'*70}")
        print(f"  BACKFILL: {date}")
        print(f"{'#'*70}")
        try:
            run_engine(backfill_date=date)
        except Exception as e:
            print(f"  ERROR on {date}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"  BACKFILL COMPLETE — {days} days processed")
    print(f"{'='*70}")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        for f in [STATE_FILE, HISTORY_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed {f}")
        print("State reset.")
        sys.exit(0)

    if "--backfill" in sys.argv:
        idx = sys.argv.index("--backfill")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 60
        backfill(n)
    else:
        run_engine()
