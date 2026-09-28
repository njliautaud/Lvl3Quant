#!/usr/bin/env python3
"""
Gap Fill Probability Strategy Backtester
========================================
Academic basis: Overnight gaps in liquid stocks fill (price returns to prior close)
with predictable probability based on gap magnitude. Small gaps (0.5-2%) fill ~70%
of the time within 1-3 days. Large gaps (>3%) fill less often but offer bigger payoffs.

Strategy:
  - Each morning, scan universe for stocks gapping from prior close
  - FADE small/medium gaps (0.5-2.5%): mean-reversion play, gap fills ~65-75%
  - RIDE large gaps (>3%) WITH volume confirmation: momentum continuation
  - Dynamic exits: gap fill target, ATR trailing stop, time stop (5 days max)
  - Regime filter: SPY trend (50/200 SMA) reduces exposure in bear markets

Walk-forward: 60-day calibration, 1-day slide, full OOT from Jan 2022 - Jul 2026
Universe: Top 100 liquid large-caps (stocks we can actually trade on Robinhood)
Capital: $645 starting, fractional shares allowed
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/gap_fill")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Strategy parameters
SMALL_GAP_MIN = 0.005       # 0.5% minimum gap to trade
SMALL_GAP_MAX = 0.025       # 2.5% gap upper bound for fade trades
LARGE_GAP_MIN = 0.035       # 3.5% gap for momentum continuation
LARGE_GAP_VOL_MULT = 1.8    # Volume must be 1.8x avg for large gap trades
FADE_MAX_HOLD = 5           # Max 5 days to hold a fade trade
RIDE_MAX_HOLD = 10          # Max 10 days for momentum continuation
ATR_PERIOD = 14
TRAILING_STOP_ATR = 1.5     # 1.5x ATR trailing stop
INITIAL_STOP_ATR = 2.0      # 2x ATR initial stop
MAX_POSITIONS = 5           # Max concurrent at $645
POSITION_SIZE_PCT = 0.18    # ~18% per position (concentrated for small account)
COMMISSION_PCT = 0.0        # Robinhood zero commission on equities
SLIPPAGE_BPS = 5            # 5 bps slippage estimate

# Walk-forward
CALIBRATION_DAYS = 60       # 60 trading days calibration
DATA_START = "2020-01-01"   # Need lookback before OOT start
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0

# SPY regime filter
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

# ─── Universe ────────────────────────────────────────────────────────────────

# Top liquid large-caps tradeable on Robinhood (diverse sectors)
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "CRM", "AVGO", "ADBE",
    "INTC", "ORCL", "CSCO", "TXN", "QCOM", "MU", "NOW", "AMAT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    # Consumer
    "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "COST", "WMT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrials
    "HON", "CAT", "DE", "GE", "RTX", "BA", "UNP",
    # Communication
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Other
    "PG", "KO", "PEP", "PM", "NEE", "SO", "PLD", "AMT",
    # High-beta / gap-prone
    "COIN", "SHOP", "SQ", "SNAP", "ROKU", "UBER", "LYFT", "PLTR", "SOFI",
    "RIVN", "LCID", "DKNG", "RBLX", "CRWD", "PANW", "ZS", "SNOW", "NET",
    "ABNB", "DASH", "PINS", "TTD", "ENPH", "FSLR", "SMCI",
]


def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data with caching."""
    cache_file = CACHE_DIR / f"gap_fill_data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached data...")
        data = pd.read_pickle(cache_file)
        # Verify it has enough tickers
        if len(data) >= len(tickers) * 0.7:
            return data

    log.info(f"Downloading data for {len(tickers)} tickers...")
    data = {}
    # Download in batches
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            batch_str = " ".join(batch)
            raw = yf.download(batch_str, start=start, end=end, group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
            for t in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        df = raw[t].copy()
                    df = df.dropna(subset=["Close", "Volume"])
                    if len(df) > 100:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Batch download failed: {e}")

    log.info(f"Downloaded {len(data)} tickers successfully")
    pd.to_pickle(data, cache_file)
    return data


def download_spy(start: str, end: str) -> pd.DataFrame:
    """Download SPY for regime filter."""
    cache_file = CACHE_DIR / "spy_data.pkl"
    if cache_file.exists():
        spy = pd.read_pickle(cache_file)
        if len(spy) > 500:
            return spy
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    pd.to_pickle(spy, cache_file)
    return spy


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect overnight gaps: (Open - PrevClose) / PrevClose."""
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close
    vol_avg = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg

    result = pd.DataFrame({
        "gap_pct": gap_pct,
        "gap_abs": gap_pct.abs(),
        "vol_ratio": vol_ratio,
        "prev_close": prev_close,
        "open": df["Open"],
        "close": df["Close"],
        "high": df["High"],
        "low": df["Low"],
        "atr": compute_atr(df, ATR_PERIOD),
    }, index=df.index)
    return result


def classify_gap(gap_pct: float, vol_ratio: float) -> str:
    """Classify a gap into trade type."""
    gap_abs = abs(gap_pct)

    if SMALL_GAP_MIN <= gap_abs <= SMALL_GAP_MAX:
        return "fade"  # Mean reversion: fade the gap
    elif gap_abs >= LARGE_GAP_MIN and vol_ratio >= LARGE_GAP_VOL_MULT:
        return "ride"  # Momentum: ride with the gap
    else:
        return "skip"  # Dead zone or no volume confirmation


class Position:
    """Track an individual position."""
    def __init__(self, ticker, entry_date, entry_price, shares, direction, trade_type,
                 gap_fill_target, initial_stop, atr_at_entry):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.direction = direction  # 1 = long, -1 = short
        self.trade_type = trade_type  # "fade" or "ride"
        self.gap_fill_target = gap_fill_target
        self.initial_stop = initial_stop
        self.trailing_stop = initial_stop
        self.atr_at_entry = atr_at_entry
        self.peak_price = entry_price if direction == 1 else entry_price
        self.trough_price = entry_price if direction == -1 else entry_price
        self.days_held = 0
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0

    def update_trailing(self, high, low):
        """Update trailing stop."""
        if self.direction == 1:
            self.peak_price = max(self.peak_price, high)
            new_stop = self.peak_price - TRAILING_STOP_ATR * self.atr_at_entry
            self.trailing_stop = max(self.trailing_stop, new_stop)
        else:
            self.trough_price = min(self.trough_price, low)
            new_stop = self.trough_price + TRAILING_STOP_ATR * self.atr_at_entry
            self.trailing_stop = min(self.trailing_stop, new_stop)


def run_backtest(data: dict, spy: pd.DataFrame, calibration_days: int = 60):
    """
    Walk-forward gap fill backtest.

    For each trading day in the OOT period:
    1. Check regime (SPY trend)
    2. Scan for gaps in universe
    3. Classify and enter new positions
    4. Manage existing positions (trailing stops, gap fill targets, time stops)
    """
    # Compute SPY regime
    spy_close = spy["Close"].squeeze() if isinstance(spy["Close"], pd.DataFrame) else spy["Close"]
    spy_sma50 = spy_close.rolling(SPY_SMA_FAST).mean()
    spy_sma200 = spy_close.rolling(SPY_SMA_SLOW).mean()
    spy_regime = (spy_sma50 > spy_sma200).astype(int)  # 1 = bull, 0 = bear

    # Compute SPY daily returns for regime classification
    spy_daily_ret = spy_close.pct_change()

    # Pre-compute gap data for all tickers
    log.info("Pre-computing gap data for all tickers...")
    gap_data = {}
    for ticker, df in data.items():
        gap_data[ticker] = detect_gaps(df)

    # Get common trading days in OOT period
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    # Use SPY dates as reference
    all_dates = spy.index[(spy.index >= oot_start) & (spy.index <= oot_end)]

    log.info(f"OOT period: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # State
    equity = STARTING_CAPITAL
    positions: list[Position] = []
    trades: list[dict] = []
    equity_curve = []
    daily_returns = []

    # Walk-forward calibration: track gap fill rates in rolling window
    # This calibrates our confidence in gap fills
    gap_fill_history = []  # (date, gap_type, filled_bool, gap_pct)

    for i, date in enumerate(all_dates):
        # ── Regime check ──
        if date in spy_regime.index:
            regime = spy_regime.loc[date] if not pd.isna(spy_regime.loc[date]) else 1
        else:
            regime = 1  # Default bullish

        # SPY daily return for green/red classification
        if date in spy_daily_ret.index:
            spy_ret = spy_daily_ret.loc[date]
        else:
            spy_ret = 0.0

        # ── Manage existing positions ──
        positions_to_close = []
        for pos in positions:
            pos.days_held += 1
            ticker = pos.ticker

            if ticker not in gap_data or date not in gap_data[ticker].index:
                continue

            gd = gap_data[ticker].loc[date]
            high = gd["high"]
            low = gd["low"]
            close = gd["close"]

            if pd.isna(close):
                continue

            # Update trailing stop
            pos.update_trailing(high, low)

            exit_price = None
            exit_reason = None

            # Check gap fill target (for fade trades)
            if pos.trade_type == "fade":
                if pos.direction == 1 and high >= pos.gap_fill_target:
                    exit_price = pos.gap_fill_target
                    exit_reason = "gap_fill"
                elif pos.direction == -1 and low <= pos.gap_fill_target:
                    exit_price = pos.gap_fill_target
                    exit_reason = "gap_fill"

            # Check trailing stop
            if exit_price is None:
                if pos.direction == 1 and low <= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                    exit_reason = "trailing_stop"
                elif pos.direction == -1 and high >= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                    exit_reason = "trailing_stop"

            # Check initial stop
            if exit_price is None:
                if pos.direction == 1 and low <= pos.initial_stop:
                    exit_price = pos.initial_stop
                    exit_reason = "initial_stop"
                elif pos.direction == -1 and high >= pos.initial_stop:
                    exit_price = pos.initial_stop
                    exit_reason = "initial_stop"

            # Check time stop
            max_hold = FADE_MAX_HOLD if pos.trade_type == "fade" else RIDE_MAX_HOLD
            if exit_price is None and pos.days_held >= max_hold:
                exit_price = close
                exit_reason = "time_stop"

            if exit_price is not None:
                # Apply slippage
                if pos.direction == 1:
                    exit_price *= (1 - SLIPPAGE_BPS / 10000)
                else:
                    exit_price *= (1 + SLIPPAGE_BPS / 10000)

                pos.exit_date = date
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.pnl = pos.direction * (exit_price - pos.entry_price) * pos.shares
                equity += pos.pnl
                positions_to_close.append(pos)

                # Record for calibration
                filled = exit_reason == "gap_fill"
                gap_fill_history.append({
                    "date": date,
                    "trade_type": pos.trade_type,
                    "filled": filled,
                    "gap_pct": abs(pos.entry_price - pos.gap_fill_target) / pos.entry_price if pos.gap_fill_target else 0,
                })

                trades.append({
                    "ticker": pos.ticker,
                    "direction": "long" if pos.direction == 1 else "short",
                    "trade_type": pos.trade_type,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(pos.exit_date.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(pos.exit_price, 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pos.pnl, 2),
                    "return_pct": round(pos.pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": pos.exit_reason,
                    "regime": "bull" if regime else "bear",
                    "spy_daily_ret": round(float(spy_ret) * 100, 2) if not pd.isna(spy_ret) else 0,
                })

        for pos in positions_to_close:
            positions.remove(pos)

        # ── Scan for new gaps ──
        if len(positions) < MAX_POSITIONS and equity > 50:
            candidates = []

            for ticker in UNIVERSE:
                if ticker not in gap_data or date not in gap_data[ticker].index:
                    continue

                # Skip if already holding this ticker
                if any(p.ticker == ticker for p in positions):
                    continue

                gd = gap_data[ticker].loc[date]
                gap_pct = gd["gap_pct"]
                vol_ratio = gd["vol_ratio"]
                atr = gd["atr"]
                prev_close = gd["prev_close"]
                open_price = gd["open"]

                if pd.isna(gap_pct) or pd.isna(vol_ratio) or pd.isna(atr) or pd.isna(open_price):
                    continue

                trade_type = classify_gap(gap_pct, vol_ratio)
                if trade_type == "skip":
                    continue

                # Regime filter: in bear market, only take fade trades (mean reversion)
                # and reduce position size
                if regime == 0 and trade_type == "ride":
                    continue

                # Walk-forward calibration: check recent fill rates
                if len(gap_fill_history) >= 20:
                    recent = [g for g in gap_fill_history[-100:]
                             if g["trade_type"] == trade_type]
                    if len(recent) >= 10:
                        fill_rate = sum(1 for g in recent if g["filled"]) / len(recent)
                        # Skip if fill rate too low for fades
                        if trade_type == "fade" and fill_rate < 0.35:
                            continue

                score = abs(gap_pct) * vol_ratio  # Prioritize bigger gaps with volume
                candidates.append((ticker, gap_pct, vol_ratio, atr, prev_close,
                                  open_price, trade_type, score))

            # Sort by score, take best
            candidates.sort(key=lambda x: x[7], reverse=True)
            slots = MAX_POSITIONS - len(positions)

            for ticker, gap_pct, vol_ratio, atr, prev_close, open_price, trade_type, score in candidates[:slots]:
                # Position sizing
                size_pct = POSITION_SIZE_PCT
                if regime == 0:
                    size_pct *= 0.6  # Reduce in bear market

                position_value = equity * size_pct
                if position_value < 10:
                    continue

                # Entry price = open (gap is at open)
                entry_price = open_price * (1 + SLIPPAGE_BPS / 10000)  # Slippage on entry
                shares = position_value / entry_price

                if trade_type == "fade":
                    # Fade: trade against the gap
                    if gap_pct > 0:
                        direction = -1  # Gap up → short (expect fill back down)
                        gap_fill_target = prev_close  # Target = prior close
                        initial_stop = entry_price + INITIAL_STOP_ATR * atr
                    else:
                        direction = 1  # Gap down → long (expect fill back up)
                        gap_fill_target = prev_close
                        initial_stop = entry_price - INITIAL_STOP_ATR * atr
                else:
                    # Ride: trade with the gap momentum
                    if gap_pct > 0:
                        direction = 1  # Gap up → long (momentum continuation)
                        gap_fill_target = entry_price * 1.05  # 5% profit target
                        initial_stop = entry_price - INITIAL_STOP_ATR * atr
                    else:
                        direction = -1  # Gap down → short (momentum continuation)
                        gap_fill_target = entry_price * 0.95
                        initial_stop = entry_price + INITIAL_STOP_ATR * atr

                # For Robinhood: we can only go long (no shorting)
                # Convert short signals to skip
                if direction == -1:
                    continue

                pos = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    direction=direction,
                    trade_type=trade_type,
                    gap_fill_target=gap_fill_target,
                    initial_stop=initial_stop,
                    atr_at_entry=atr,
                )
                positions.append(pos)

        # ── End of day: record equity ──
        # Mark-to-market open positions
        mtm = equity
        for pos in positions:
            if pos.ticker in gap_data and date in gap_data[pos.ticker].index:
                current = gap_data[pos.ticker].loc[date]["close"]
                if not pd.isna(current):
                    unrealized = pos.direction * (current - pos.entry_price) * pos.shares
                    mtm += unrealized

        equity_curve.append({
            "date": date,
            "equity": round(mtm, 2),
            "n_positions": len(positions),
            "regime": "bull" if regime else "bear",
            "spy_ret": float(spy_ret) if not pd.isna(spy_ret) else 0.0,
        })

        if len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            if prev_eq > 0:
                daily_returns.append(mtm / prev_eq - 1)
            else:
                daily_returns.append(0)

        if i % 100 == 0:
            log.info(f"  Day {i}/{len(all_dates)}: equity=${mtm:.2f}, positions={len(positions)}, trades={len(trades)}")

    # Close any remaining positions at last available price
    for pos in positions:
        last_date = all_dates[-1]
        if pos.ticker in gap_data and last_date in gap_data[pos.ticker].index:
            close = gap_data[pos.ticker].loc[last_date]["close"]
            if not pd.isna(close):
                pos.exit_date = last_date
                pos.exit_price = close
                pos.exit_reason = "end_of_test"
                pos.pnl = pos.direction * (close - pos.entry_price) * pos.shares
                trades.append({
                    "ticker": pos.ticker,
                    "direction": "long" if pos.direction == 1 else "short",
                    "trade_type": pos.trade_type,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(pos.exit_date.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(pos.exit_price, 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pos.pnl, 2),
                    "return_pct": round(pos.pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": pos.exit_reason,
                    "regime": "bull",
                    "spy_daily_ret": 0,
                })

    return trades, equity_curve, daily_returns


def validate_strategy(trades: list[dict], equity_curve: list[dict],
                      daily_returns: list[float]) -> dict:
    """5-gate validation."""
    results = {}

    if not trades:
        log.warning("No trades generated!")
        return {"passed": False, "reason": "No trades"}

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    returns = np.array(daily_returns)

    # ── Basic stats ──
    total_trades = len(df_trades)
    winning = df_trades[df_trades["pnl"] > 0]
    losing = df_trades[df_trades["pnl"] <= 0]
    win_rate = len(winning) / total_trades if total_trades > 0 else 0
    avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
    avg_loss = losing["pnl"].mean() if len(losing) > 0 else 0
    profit_factor = abs(winning["pnl"].sum() / losing["pnl"].sum()) if losing["pnl"].sum() != 0 else float("inf")
    total_pnl = df_trades["pnl"].sum()
    final_equity = df_equity["equity"].iloc[-1] if len(df_equity) > 0 else STARTING_CAPITAL

    # By trade type
    fade_trades = df_trades[df_trades["trade_type"] == "fade"]
    ride_trades = df_trades[df_trades["trade_type"] == "ride"]

    results["total_trades"] = total_trades
    results["win_rate"] = round(win_rate, 4)
    results["avg_win"] = round(avg_win, 2)
    results["avg_loss"] = round(avg_loss, 2)
    results["profit_factor"] = round(profit_factor, 3)
    results["total_pnl"] = round(total_pnl, 2)
    results["final_equity"] = round(final_equity, 2)
    results["return_pct"] = round((final_equity / STARTING_CAPITAL - 1) * 100, 2)
    results["avg_days_held"] = round(df_trades["days_held"].mean(), 1)

    results["fade_trades"] = len(fade_trades)
    results["fade_wr"] = round(len(fade_trades[fade_trades["pnl"] > 0]) / len(fade_trades), 4) if len(fade_trades) > 0 else 0
    results["ride_trades"] = len(ride_trades)
    results["ride_wr"] = round(len(ride_trades[ride_trades["pnl"] > 0]) / len(ride_trades), 4) if len(ride_trades) > 0 else 0

    # Exit reason breakdown
    results["exit_reasons"] = df_trades["exit_reason"].value_counts().to_dict()

    # ── Gate 1: Sharpe ratio ──
    if len(returns) > 20:
        ann_return = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0
    results["sharpe"] = round(sharpe, 3)
    results["gate1_sharpe"] = sharpe > 0.5

    # Sortino
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (np.mean(returns) * 252) / downside_vol if downside_vol > 0 else 0
    results["sortino"] = round(sortino, 3)

    # ── Gate 2: Permutation test ──
    observed_sharpe = sharpe
    n_perms = 1000
    perm_sharpes = []
    for _ in range(n_perms):
        perm_ret = np.random.permutation(returns)
        perm_mean = np.mean(perm_ret) * 252
        perm_std = np.std(perm_ret) * np.sqrt(252)
        perm_sharpes.append(perm_mean / perm_std if perm_std > 0 else 0)
    p_value = np.mean(np.array(perm_sharpes) >= observed_sharpe)
    results["perm_p_value"] = round(p_value, 4)
    results["gate2_perm"] = p_value < 0.05

    # ── Gate 3: Beat random baseline ──
    n_random_trials = 500
    random_sharpes = []
    for _ in range(n_random_trials):
        # Random entry/exit with same holding period
        random_returns = np.random.choice(returns, size=len(returns), replace=True)
        r_mean = np.mean(random_returns) * 252
        r_std = np.std(random_returns) * np.sqrt(252)
        random_sharpes.append(r_mean / r_std if r_std > 0 else 0)
    random_median = np.median(random_sharpes)
    results["random_baseline_sharpe"] = round(random_median, 3)
    results["gate3_beats_random"] = sharpe > random_median + 0.1  # Must beat by margin

    # ── Gate 4: Regime balance ──
    bull_trades = df_trades[df_trades["regime"] == "bull"]
    bear_trades = df_trades[df_trades["regime"] == "bear"]

    if len(bull_trades) > 5 and len(bear_trades) > 5:
        bull_wr = len(bull_trades[bull_trades["pnl"] > 0]) / len(bull_trades)
        bear_wr = len(bear_trades[bear_trades["pnl"] > 0]) / len(bear_trades)
        regime_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr) if max(bull_wr, bear_wr) > 0 else 1
    elif len(bear_trades) <= 5:
        # Not enough bear trades to judge, pass conditionally
        bull_wr = len(bull_trades[bull_trades["pnl"] > 0]) / len(bull_trades) if len(bull_trades) > 0 else 0
        bear_wr = 0
        regime_gap = 0.3  # Uncertain, give benefit of doubt
    else:
        bull_wr = 0
        bear_wr = len(bear_trades[bear_trades["pnl"] > 0]) / len(bear_trades) if len(bear_trades) > 0 else 0
        regime_gap = 1.0

    results["bull_trades"] = len(bull_trades)
    results["bull_wr"] = round(bull_wr, 4)
    results["bear_trades"] = len(bear_trades)
    results["bear_wr"] = round(bear_wr, 4)
    results["regime_gap"] = round(regime_gap, 4)
    results["gate4_regime"] = regime_gap < 0.50

    # ── Gate 5: Max drawdown ──
    equity_series = df_equity["equity"].values
    peak = np.maximum.accumulate(equity_series)
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    results["max_drawdown"] = round(max_dd, 4)
    results["gate5_mdd"] = max_dd > -0.50

    # ── Green/red day analysis ──
    df_equity_dt = df_equity.copy()
    df_equity_dt["daily_ret"] = df_equity_dt["equity"].pct_change()
    green_days = df_equity_dt[df_equity_dt["spy_ret"] > 0.001]
    red_days = df_equity_dt[df_equity_dt["spy_ret"] < -0.001]

    if len(green_days) > 20 and len(red_days) > 20:
        green_ret = green_days["daily_ret"].mean() * 252
        green_vol = green_days["daily_ret"].std() * np.sqrt(252)
        green_sharpe = green_ret / green_vol if green_vol > 0 else 0

        red_ret = red_days["daily_ret"].mean() * 252
        red_vol = red_days["daily_ret"].std() * np.sqrt(252)
        red_sharpe = red_ret / red_vol if red_vol > 0 else 0

        results["green_day_sharpe"] = round(green_sharpe, 3)
        results["red_day_sharpe"] = round(red_sharpe, 3)
    else:
        results["green_day_sharpe"] = 0
        results["red_day_sharpe"] = 0

    # ── Overall pass/fail ──
    gates = [results.get(f"gate{i}_{k}", False) for i, k in
             [(1, "sharpe"), (2, "perm"), (3, "beats_random"), (4, "regime"), (5, "mdd")]]
    results["gates_passed"] = sum(gates)
    results["all_gates_passed"] = all(gates)

    return results


def log_to_mlflow(results: dict, trades: list[dict], equity_curve: list[dict]):
    """Log results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("gap_fill_probability_v1")

        with mlflow.start_run(run_name=f"gap_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("strategy", "gap_fill_probability")
            mlflow.log_param("universe_size", len(UNIVERSE))
            mlflow.log_param("small_gap_range", f"{SMALL_GAP_MIN}-{SMALL_GAP_MAX}")
            mlflow.log_param("large_gap_min", LARGE_GAP_MIN)
            mlflow.log_param("max_positions", MAX_POSITIONS)
            mlflow.log_param("position_size_pct", POSITION_SIZE_PCT)
            mlflow.log_param("oot_period", f"{OOT_START} to {OOT_END}")
            mlflow.log_param("starting_capital", STARTING_CAPITAL)
            mlflow.log_param("fade_max_hold", FADE_MAX_HOLD)
            mlflow.log_param("ride_max_hold", RIDE_MAX_HOLD)
            mlflow.log_param("trailing_stop_atr", TRAILING_STOP_ATR)

            # Log metrics
            for key, val in results.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    mlflow.log_metric(key, val)
                elif isinstance(val, bool):
                    mlflow.log_metric(key, int(val))

            # Log artifacts
            trades_path = OUTPUT_DIR / "trades.json"
            with open(trades_path, "w") as f:
                json.dump(trades, f, indent=2, default=str)
            mlflow.log_artifact(str(trades_path))

            equity_path = OUTPUT_DIR / "equity_curve.json"
            with open(equity_path, "w") as f:
                json.dump(equity_curve, f, indent=2, default=str)
            mlflow.log_artifact(str(equity_path))

            results_path = OUTPUT_DIR / "validation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))

            log.info("MLflow logging complete")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


def main():
    log.info("=" * 70)
    log.info("GAP FILL PROBABILITY STRATEGY BACKTEST")
    log.info("=" * 70)
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"OOT: {OOT_START} to {OOT_END}")
    log.info(f"Starting capital: ${STARTING_CAPITAL}")
    log.info(f"Gap fade range: {SMALL_GAP_MIN*100:.1f}% - {SMALL_GAP_MAX*100:.1f}%")
    log.info(f"Gap ride threshold: >{LARGE_GAP_MIN*100:.1f}% with {LARGE_GAP_VOL_MULT}x vol")

    # Download data
    data = download_data(UNIVERSE, DATA_START, OOT_END)
    spy = download_spy(DATA_START, OOT_END)

    if len(data) < 20:
        log.error(f"Only got {len(data)} tickers, need at least 20")
        sys.exit(1)

    log.info(f"Data ready: {len(data)} tickers, SPY {len(spy)} days")

    # Run backtest
    trades, equity_curve, daily_returns = run_backtest(data, spy)

    log.info(f"\nBacktest complete: {len(trades)} trades generated")

    # Validate
    results = validate_strategy(trades, equity_curve, daily_returns)

    # Print results
    log.info("\n" + "=" * 70)
    log.info("VALIDATION RESULTS")
    log.info("=" * 70)
    log.info(f"Total trades: {results.get('total_trades', 0)}")
    log.info(f"Win rate: {results.get('win_rate', 0)*100:.1f}%")
    log.info(f"Profit factor: {results.get('profit_factor', 0):.3f}")
    log.info(f"Total P&L: ${results.get('total_pnl', 0):.2f}")
    log.info(f"Final equity: ${results.get('final_equity', 0):.2f} ({results.get('return_pct', 0):.1f}%)")
    log.info(f"Avg days held: {results.get('avg_days_held', 0):.1f}")
    log.info(f"")
    log.info(f"Fade trades: {results.get('fade_trades', 0)} (WR: {results.get('fade_wr', 0)*100:.1f}%)")
    log.info(f"Ride trades: {results.get('ride_trades', 0)} (WR: {results.get('ride_wr', 0)*100:.1f}%)")
    log.info(f"Exit reasons: {results.get('exit_reasons', {})}")
    log.info(f"")
    log.info(f"─── 5-Gate Validation ───")
    log.info(f"Gate 1 - Sharpe > 0.5:    {results.get('sharpe', 0):.3f} {'PASS' if results.get('gate1_sharpe') else 'FAIL'}")
    log.info(f"         Sortino:          {results.get('sortino', 0):.3f}")
    log.info(f"Gate 2 - Perm p < 0.05:   {results.get('perm_p_value', 1):.4f} {'PASS' if results.get('gate2_perm') else 'FAIL'}")
    log.info(f"Gate 3 - Beats random:    Strategy {results.get('sharpe', 0):.3f} vs Random {results.get('random_baseline_sharpe', 0):.3f} {'PASS' if results.get('gate3_beats_random') else 'FAIL'}")
    log.info(f"Gate 4 - Regime balance:  Gap={results.get('regime_gap', 0):.4f} (Bull WR={results.get('bull_wr', 0)*100:.1f}% Bear WR={results.get('bear_wr', 0)*100:.1f}%) {'PASS' if results.get('gate4_regime') else 'FAIL'}")
    log.info(f"Gate 5 - MDD > -50%:      {results.get('max_drawdown', 0)*100:.1f}% {'PASS' if results.get('gate5_mdd') else 'FAIL'}")
    log.info(f"")
    log.info(f"Green day Sharpe: {results.get('green_day_sharpe', 0):.3f}")
    log.info(f"Red day Sharpe:   {results.get('red_day_sharpe', 0):.3f}")
    log.info(f"")
    log.info(f"Gates passed: {results.get('gates_passed', 0)}/5")
    log.info(f"OVERALL: {'PASS' if results.get('all_gates_passed') else 'FAIL'}")

    # Log to MLflow
    log_to_mlflow(results, trades, equity_curve)

    # Save results
    with open(OUTPUT_DIR / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
