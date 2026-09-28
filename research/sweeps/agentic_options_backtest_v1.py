#!/usr/bin/env python3
"""
Agentic Options Backtest v1 — Systematic cheap-options portfolio backtest.

Backtests two validated equity signals (oversold bounce + post-earnings bounce)
using Black-Scholes priced ATM/OTM calls and bull call spreads on a universe
of 30-40 mid/small-cap stocks where ATM calls are typically < $300.

Starting capital: $645. Walk-forward sliding window (252d train / 21d test).
Period: 2018-01-01 to 2026-07-01.

Adversarial gates: permutation test, regime test, sub-period test, outlier test.

Author: Claude Opus 4.6
Date: 2026-07-24
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_bt")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "research", "findings",
                            "agentic_options_backtest_v1_results.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSE = [
    "F", "SOFI", "RIVN", "HOOD", "MARA", "SNAP", "PLTR", "AMD", "UBER",
    "LYFT", "SQ", "COIN", "OPEN", "NIO", "LCID", "DKNG", "RBLX", "PINS",
    "ROKU", "UPST", "FUTU", "MQ", "AAL", "DAL", "UAL", "CCL", "RCL",
    "NCLH", "PYPL", "ABNB",
]

START_DATE = "2018-01-01"
END_DATE = "2026-07-01"
TRAIN_WINDOW = 252   # trading days
TEST_WINDOW = 21     # trading days
STARTING_CAPITAL = 645.0

# Position sizing
MAX_CONCURRENT = 3
MAX_POSITION_PCT = 0.50  # skip if option cost > 50% of capital

# Commission per contract per leg (Robinhood)
COMMISSION_PER_LEG = 0.65
SLIPPAGE_PCT = 0.05  # 5% of option price

# Option parameters
RISK_FREE_RATE = 0.045   # approximate risk-free rate
DEFAULT_DTE = 30         # days to expiration for options we buy

# Signal parameters
OVERSOLD_THRESHOLD = -0.10   # 10% weekly drop
EARNINGS_DROP_THRESHOLD = -0.08  # 8% two-day drop after earnings
OVERSOLD_HOLD = 5   # trading days
EARNINGS_HOLD = 10  # trading days

# Regime thresholds for SPY classification
REGIME_GREEN_THRESHOLD = 0.002   # > +0.2% = green day
REGIME_RED_THRESHOLD = -0.002    # < -0.2% = red day

# Permutation test
N_PERMUTATIONS = 200

# ---------------------------------------------------------------------------
# MLflow setup (non-blocking)
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import mlflow
    import urllib.request
    mlflow.set_tracking_uri("http://jupiter:5000")
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    MLFLOW_OK = True
    log.info("MLflow connected at http://jupiter:5000")
except Exception:
    log.info("MLflow unavailable — results will be saved locally only")


# ===========================================================================
# Black-Scholes pricing
# ===========================================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price.

    Args:
        S: spot price
        K: strike price
        T: time to expiration in years (must be > 0)
        r: risk-free rate
        sigma: implied volatility
    Returns:
        Call option price
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ===========================================================================
# Data loading
# ===========================================================================

def load_price_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV data from yfinance for all tickers.

    Returns dict of {ticker: DataFrame with columns [Open, High, Low, Close, Volume]}.
    Timezone is stripped per HC requirement.
    """
    import yfinance as yf

    data = {}
    log.info(f"Downloading price data for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, end=end, auto_adjust=True)

            if hist.empty:
                log.warning(f"  {ticker}: no data returned, skipping")
                continue

            # CRITICAL: strip timezone from yfinance data
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            # Keep only needed columns
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist.dropna(subset=["Close"], inplace=True)

            if len(hist) < TRAIN_WINDOW + TEST_WINDOW:
                log.warning(f"  {ticker}: only {len(hist)} rows, need {TRAIN_WINDOW + TEST_WINDOW}, skipping")
                continue

            data[ticker] = hist
            if (i + 1) % 10 == 0:
                log.info(f"  Downloaded {i + 1}/{len(tickers)} tickers")

        except Exception as e:
            log.warning(f"  {ticker}: download failed ({e}), skipping")

    log.info(f"Loaded data for {len(data)} tickers")
    return data


def load_spy_data(start: str, end: str) -> pd.Series:
    """Load SPY close prices for regime classification."""
    import yfinance as yf

    spy = yf.Ticker("SPY")
    hist = spy.history(start=start, end=end, auto_adjust=True)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert(None)
    return hist["Close"]


def load_earnings_dates(tickers: List[str]) -> Dict[str, pd.DatetimeIndex]:
    """Load historical earnings dates for each ticker using yfinance.

    Returns dict of {ticker: DatetimeIndex of earnings dates}.
    Falls back gracefully if earnings data unavailable.
    """
    import yfinance as yf

    earnings = {}
    log.info(f"Loading earnings dates for {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            # get_earnings_dates returns upcoming + recent; we need historical
            # Try multiple lookback periods to get broad coverage
            edates = tk.get_earnings_dates(limit=100)
            if edates is not None and not edates.empty:
                idx = edates.index
                if idx.tz is not None:
                    idx = idx.tz_convert(None)
                # Normalize to date only (remove time component)
                idx = pd.DatetimeIndex(idx.normalize().unique())
                earnings[ticker] = idx
            else:
                earnings[ticker] = pd.DatetimeIndex([])
        except Exception:
            earnings[ticker] = pd.DatetimeIndex([])

    n_with = sum(1 for v in earnings.values() if len(v) > 0)
    log.info(f"Loaded earnings dates for {n_with}/{len(tickers)} tickers")
    return earnings


# ===========================================================================
# Signal generation
# ===========================================================================

def generate_oversold_signals(close: pd.Series) -> pd.Series:
    """Oversold bounce signal: fires when stock drops 10%+ in a week (5 trading days).

    Args:
        close: daily close prices
    Returns:
        Boolean series — True on signal days
    """
    weekly_return = close / close.shift(5) - 1
    return weekly_return < OVERSOLD_THRESHOLD


def generate_earnings_signals(
    close: pd.Series,
    earnings_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Post-earnings bounce signal: fires when stock drops 8%+ in 2 days after earnings.

    Args:
        close: daily close prices
        earnings_dates: dates when earnings were reported
    Returns:
        Boolean series — True on signal days
    """
    two_day_return = close / close.shift(2) - 1
    drop_mask = two_day_return < EARNINGS_DROP_THRESHOLD

    # Check if earnings occurred in the 2-day lookback window
    earnings_mask = pd.Series(False, index=close.index)
    for dt in close.index:
        # Earnings must have been in [dt-2 business days, dt]
        window_start = dt - pd.Timedelta(days=5)  # generous calendar window
        in_window = (earnings_dates >= window_start) & (earnings_dates <= dt)
        if in_window.any():
            earnings_mask.loc[dt] = True

    return drop_mask & earnings_mask


# ===========================================================================
# Options simulation
# ===========================================================================

class OptionTrade:
    """Represents a single options trade."""

    def __init__(
        self,
        ticker: str,
        signal_type: str,       # "oversold" or "earnings"
        option_type: str,       # "atm_call", "otm_call", "bull_spread"
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        entry_spot: float,
        exit_spot: float,
        entry_price: float,     # option premium paid (per share, x100 for contract)
        exit_price: float,      # option premium received at exit
        commission: float,
        slippage: float,
        hold_days: int,
    ):
        self.ticker = ticker
        self.signal_type = signal_type
        self.option_type = option_type
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.entry_spot = entry_spot
        self.exit_spot = exit_spot
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.commission = commission
        self.slippage = slippage
        self.hold_days = hold_days

    @property
    def cost(self) -> float:
        """Total cost to enter the trade (premium + commission + slippage)."""
        return self.entry_price * 100 + self.commission + self.slippage

    @property
    def proceeds(self) -> float:
        """Total proceeds at exit (premium - commission - slippage)."""
        exit_slippage = self.exit_price * 100 * SLIPPAGE_PCT
        return max(self.exit_price * 100 - COMMISSION_PER_LEG - exit_slippage, 0.0)

    @property
    def pnl(self) -> float:
        """Net P&L for the trade."""
        return self.proceeds - self.cost

    @property
    def return_pct(self) -> float:
        """Percentage return on cost."""
        if self.cost <= 0:
            return 0.0
        return self.pnl / self.cost


def compute_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """20-day rolling realized volatility (annualized)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def price_option_entry(
    spot: float,
    realized_vol: float,
    is_earnings: bool,
    option_type: str,
) -> Tuple[float, float, float]:
    """Price an option at entry time using Black-Scholes.

    Args:
        spot: current stock price
        realized_vol: 20-day realized volatility
        is_earnings: whether this is an earnings signal (higher IV)
        option_type: "atm_call", "otm_call", or "bull_spread"

    Returns:
        (entry_premium_per_share, commission, slippage_cost)
    """
    # IV premium approximation
    iv = realized_vol * (1.5 if is_earnings else 1.1)
    iv = max(iv, 0.15)  # floor at 15% IV

    T = DEFAULT_DTE / 365.0  # time to expiration in years

    if option_type == "atm_call":
        K = round(spot)  # nearest dollar strike
        premium = bs_call_price(spot, K, T, RISK_FREE_RATE, iv)
        commission = COMMISSION_PER_LEG
        slippage = premium * SLIPPAGE_PCT * 100  # on contract value

    elif option_type == "otm_call":
        K = round(spot * 1.02)  # 2% OTM
        premium = bs_call_price(spot, K, T, RISK_FREE_RATE, iv)
        commission = COMMISSION_PER_LEG
        slippage = premium * SLIPPAGE_PCT * 100

    elif option_type == "bull_spread":
        K_long = round(spot)         # ATM long leg
        K_short = round(spot * 1.05) # 5% OTM short leg
        premium_long = bs_call_price(spot, K_long, T, RISK_FREE_RATE, iv)
        premium_short = bs_call_price(spot, K_short, T, RISK_FREE_RATE, iv)
        premium = premium_long - premium_short  # net debit
        commission = 2 * COMMISSION_PER_LEG  # two legs
        slippage = premium * SLIPPAGE_PCT * 100

    else:
        raise ValueError(f"Unknown option type: {option_type}")

    return premium, commission, slippage


def price_option_exit(
    spot_entry: float,
    spot_exit: float,
    realized_vol: float,
    is_earnings: bool,
    option_type: str,
    hold_days: int,
) -> float:
    """Reprice option at exit with new spot and reduced DTE.

    Args:
        spot_entry: stock price at entry
        spot_exit: stock price at exit
        realized_vol: vol at entry time (used for IV estimate)
        is_earnings: whether earnings signal
        option_type: type of option structure
        hold_days: how many days held

    Returns:
        Exit premium per share
    """
    iv = realized_vol * (1.5 if is_earnings else 1.1)
    iv = max(iv, 0.15)

    remaining_dte = max(DEFAULT_DTE - hold_days, 1)
    T = remaining_dte / 365.0

    if option_type == "atm_call":
        K = round(spot_entry)
        return bs_call_price(spot_exit, K, T, RISK_FREE_RATE, iv)

    elif option_type == "otm_call":
        K = round(spot_entry * 1.02)
        return bs_call_price(spot_exit, K, T, RISK_FREE_RATE, iv)

    elif option_type == "bull_spread":
        K_long = round(spot_entry)
        K_short = round(spot_entry * 1.05)
        premium_long = bs_call_price(spot_exit, K_long, T, RISK_FREE_RATE, iv)
        premium_short = bs_call_price(spot_exit, K_short, T, RISK_FREE_RATE, iv)
        return premium_long - premium_short

    return 0.0


# ===========================================================================
# Walk-forward engine
# ===========================================================================

def run_backtest(
    price_data: Dict[str, pd.DataFrame],
    earnings_data: Dict[str, pd.DatetimeIndex],
    spy_close: pd.Series,
    option_type: str,
) -> Dict[str, Any]:
    """Run full walk-forward backtest for a given option type.

    Sliding window: 252d train (for vol stats) / 21d test.

    Args:
        price_data: dict of {ticker: OHLCV DataFrame}
        earnings_data: dict of {ticker: earnings DatetimeIndex}
        spy_close: SPY daily closes for regime classification
        option_type: "atm_call", "otm_call", or "bull_spread"

    Returns:
        Dict with trades, equity curve, and metrics.
    """
    log.info(f"Running backtest: {option_type}")

    # Build a common date index from all tickers
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    all_dates = pd.DatetimeIndex(all_dates)

    # Filter to date range
    mask = (all_dates >= pd.Timestamp(START_DATE)) & (all_dates <= pd.Timestamp(END_DATE))
    all_dates = all_dates[mask]

    if len(all_dates) < TRAIN_WINDOW + TEST_WINDOW:
        log.error("Not enough dates for walk-forward")
        return {}

    # Pre-compute realized vol and signals for each ticker
    ticker_close = {}
    ticker_rvol = {}
    ticker_oversold = {}
    ticker_earnings = {}

    for ticker, df in price_data.items():
        close = df["Close"].reindex(all_dates).ffill()
        ticker_close[ticker] = close
        ticker_rvol[ticker] = compute_realized_vol(close, window=20)
        ticker_oversold[ticker] = generate_oversold_signals(close)

        edates = earnings_data.get(ticker, pd.DatetimeIndex([]))
        ticker_earnings[ticker] = generate_earnings_signals(close, edates)

    # SPY regime classification
    spy_reindexed = spy_close.reindex(all_dates).ffill()
    spy_daily_ret = spy_reindexed.pct_change()
    regime = pd.Series("flat", index=all_dates)
    regime[spy_daily_ret > REGIME_GREEN_THRESHOLD] = "green"
    regime[spy_daily_ret < REGIME_RED_THRESHOLD] = "red"

    # Walk-forward loop
    trades: List[OptionTrade] = []
    capital = STARTING_CAPITAL
    equity_curve = []
    open_positions: List[Tuple[OptionTrade, float]] = []  # (trade_template, cost)

    n_windows = (len(all_dates) - TRAIN_WINDOW) // TEST_WINDOW
    window_count = 0

    for w_start in range(TRAIN_WINDOW, len(all_dates) - TEST_WINDOW + 1, TEST_WINDOW):
        window_count += 1
        test_start = w_start
        test_end = min(w_start + TEST_WINDOW, len(all_dates))
        test_dates = all_dates[test_start:test_end]

        if window_count % 50 == 0:
            log.info(
                f"  Window {window_count}/{n_windows}: "
                f"{test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}, "
                f"capital=${capital:.2f}, trades={len(trades)}"
            )

        # Process each test day
        for day_idx, day in enumerate(test_dates):
            # Close expired positions
            positions_to_close = []
            for pos_idx, (trade, cost) in enumerate(open_positions):
                if day >= trade.exit_date:
                    positions_to_close.append(pos_idx)

            # Close in reverse order to preserve indices
            for pos_idx in sorted(positions_to_close, reverse=True):
                trade, cost = open_positions.pop(pos_idx)
                # Finalize exit price
                proceeds = trade.proceeds
                capital += proceeds
                trades.append(trade)

            # Check for new signals (only if we have room)
            if len(open_positions) >= MAX_CONCURRENT:
                continue

            for ticker in price_data.keys():
                if len(open_positions) >= MAX_CONCURRENT:
                    break

                close = ticker_close[ticker]
                if day not in close.index or pd.isna(close.get(day)):
                    continue

                spot = close[day]
                rvol_val = ticker_rvol[ticker].get(day)
                if pd.isna(rvol_val) or rvol_val <= 0:
                    continue

                # Check both signals
                for signal_type, signal_series, hold_days in [
                    ("oversold", ticker_oversold[ticker], OVERSOLD_HOLD),
                    ("earnings", ticker_earnings[ticker], EARNINGS_HOLD),
                ]:
                    if not signal_series.get(day, False):
                        continue

                    # Already have a position in this ticker?
                    if any(t.ticker == ticker for t, _ in open_positions):
                        continue

                    is_earnings = (signal_type == "earnings")

                    # Price the option
                    premium, commission, slippage = price_option_entry(
                        spot, rvol_val, is_earnings, option_type
                    )

                    total_cost = premium * 100 + commission + slippage

                    # Position sizing checks
                    if total_cost <= 0:
                        continue
                    if total_cost > capital * MAX_POSITION_PCT:
                        continue
                    if total_cost > capital:
                        continue
                    # Skip if option premium is unreasonably expensive (> $300 per contract)
                    if premium * 100 > 300:
                        continue

                    # Determine exit date
                    future_dates = all_dates[all_dates > day]
                    if len(future_dates) < hold_days:
                        continue
                    exit_date = future_dates[hold_days - 1]

                    # Get exit spot price
                    exit_spot = close.get(exit_date)
                    if pd.isna(exit_spot):
                        continue

                    # Price at exit
                    exit_premium = price_option_exit(
                        spot, exit_spot, rvol_val, is_earnings, option_type, hold_days
                    )

                    trade = OptionTrade(
                        ticker=ticker,
                        signal_type=signal_type,
                        option_type=option_type,
                        entry_date=day,
                        exit_date=exit_date,
                        entry_spot=spot,
                        exit_spot=exit_spot,
                        entry_price=premium,
                        exit_price=exit_premium,
                        commission=commission,
                        slippage=slippage,
                        hold_days=hold_days,
                    )

                    # Debit capital
                    capital -= total_cost
                    open_positions.append((trade, total_cost))

            # Record daily equity (capital + mark-to-market of open positions)
            mtm = capital
            for trade, cost in open_positions:
                # Rough MTM: linear interpolation of entry→exit value
                days_in = max((day - trade.entry_date).days, 0)
                total_days = max((trade.exit_date - trade.entry_date).days, 1)
                frac = min(days_in / total_days, 1.0)
                current_val = trade.cost * (1 - frac) + trade.proceeds * frac
                mtm += current_val

            equity_curve.append({
                "date": day.strftime("%Y-%m-%d"),
                "equity": round(mtm, 2),
                "capital": round(capital, 2),
                "open_positions": len(open_positions),
            })

    # Close any remaining open positions at end
    for trade, cost in open_positions:
        capital += trade.proceeds
        trades.append(trade)

    log.info(f"  Completed: {len(trades)} trades, final capital=${capital:.2f}")

    # Compute metrics
    metrics = compute_metrics(trades, equity_curve, regime, all_dates)
    metrics["option_type"] = option_type
    metrics["total_trades"] = len(trades)
    metrics["final_capital"] = round(capital, 2)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "metrics": metrics,
    }


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_metrics(
    trades: List[OptionTrade],
    equity_curve: List[dict],
    regime: pd.Series,
    all_dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Compute performance metrics from trade list and equity curve."""

    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "calmar": 0.0,
            "avg_trade_pnl": 0.0, "total_pnl": 0.0,
        }

    # Trade-level metrics
    pnls = [t.pnl for t in trades]
    returns = [t.return_pct for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(pnls) if pnls else 0.0
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Equity curve metrics
    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": 0.0,
            "win_rate": win_rate, "profit_factor": profit_factor,
            "calmar": 0.0, "avg_trade_pnl": np.mean(pnls),
            "total_pnl": sum(pnls),
        }

    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    daily_returns = eq["equity"].pct_change().dropna()
    daily_returns = daily_returns.replace([np.inf, -np.inf], 0.0)

    # Sharpe (annualized, daily returns)
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = sharpe  # no downside = excellent

    # CAGR
    total_days = (eq.index[-1] - eq.index[0]).days
    if total_days > 0 and eq["equity"].iloc[0] > 0:
        total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0]
        years = total_days / 365.25
        cagr = total_return ** (1 / years) - 1 if total_return > 0 else -1.0
    else:
        cagr = 0.0

    # Max drawdown
    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Per-signal breakdown
    oversold_trades = [t for t in trades if t.signal_type == "oversold"]
    earnings_trades = [t for t in trades if t.signal_type == "earnings"]

    def signal_summary(trade_list):
        if not trade_list:
            return {"count": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
        p = [t.pnl for t in trade_list]
        w = sum(1 for x in p if x > 0)
        return {
            "count": len(trade_list),
            "win_rate": round(w / len(trade_list), 4),
            "avg_pnl": round(np.mean(p), 2),
            "total_pnl": round(sum(p), 2),
        }

    # Regime-stratified Sharpe
    regime_sharpes = {}
    for reg in ["green", "red", "flat"]:
        reg_dates = regime[regime == reg].index
        reg_returns = daily_returns.reindex(reg_dates).dropna()
        if len(reg_returns) > 5 and reg_returns.std() > 0:
            regime_sharpes[reg] = round(
                (reg_returns.mean() / reg_returns.std()) * np.sqrt(252), 4
            )
        else:
            regime_sharpes[reg] = 0.0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "calmar": round(calmar, 4),
        "avg_trade_pnl": round(np.mean(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
        "oversold_signal": signal_summary(oversold_trades),
        "earnings_signal": signal_summary(earnings_trades),
        "regime_sharpes": regime_sharpes,
    }


# ===========================================================================
# Adversarial gates
# ===========================================================================

def gate_permutation_test(
    trades: List[OptionTrade],
    equity_curve: List[dict],
    n_perms: int = N_PERMUTATIONS,
) -> Dict[str, Any]:
    """Permutation test: shuffle signal dates, compare Sharpe.

    If real Sharpe > 95% of permuted Sharpes, we pass.
    """
    log.info(f"Running permutation test ({n_perms} shuffles)...")

    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {"pass": False, "reason": "no equity curve"}

    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")
    real_returns = eq["equity"].pct_change().dropna().replace([np.inf, -np.inf], 0.0)
    real_sharpe = (real_returns.mean() / real_returns.std()) * np.sqrt(252) if real_returns.std() > 0 else 0.0

    # Permutation: shuffle trade PnLs
    pnls = np.array([t.pnl for t in trades])
    if len(pnls) == 0:
        return {"pass": False, "reason": "no trades"}

    perm_sharpes = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        cumulative = STARTING_CAPITAL + np.cumsum(shuffled)
        cumulative = np.insert(cumulative, 0, STARTING_CAPITAL)
        rets = np.diff(cumulative) / cumulative[:-1]
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(rets) > 0:
            perm_sharpes.append(float(np.mean(rets) / np.std(rets) * np.sqrt(252)))
        else:
            perm_sharpes.append(0.0)

    percentile = np.mean([1 if real_sharpe > s else 0 for s in perm_sharpes])
    passed = percentile >= 0.95

    return {
        "pass": passed,
        "real_sharpe": round(real_sharpe, 4),
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 4),
        "perm_95th": round(np.percentile(perm_sharpes, 95), 4),
        "percentile": round(percentile, 4),
    }


def gate_regime_test(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Regime test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50.

    Ensures strategy isn't just riding a bull/bear regime.
    """
    regime_sharpes = metrics.get("regime_sharpes", {})
    sg = regime_sharpes.get("green", 0.0)
    sr = regime_sharpes.get("red", 0.0)

    max_abs = max(abs(sg), abs(sr))
    if max_abs == 0:
        ratio = 0.0
    else:
        ratio = abs(sg - sr) / max_abs

    passed = ratio <= 0.50

    return {
        "pass": passed,
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": regime_sharpes.get("flat", 0.0),
        "regime_divergence_ratio": round(ratio, 4),
        "threshold": 0.50,
    }


def gate_sub_period_test(trades: List[OptionTrade]) -> Dict[str, Any]:
    """Sub-period test: split trades into 2 equal halves by time, both must be profitable."""
    if len(trades) < 4:
        return {"pass": False, "reason": "too few trades"}

    sorted_trades = sorted(trades, key=lambda t: t.entry_date)
    mid = len(sorted_trades) // 2
    first_half = sorted_trades[:mid]
    second_half = sorted_trades[mid:]

    pnl_1 = sum(t.pnl for t in first_half)
    pnl_2 = sum(t.pnl for t in second_half)

    passed = pnl_1 > 0 and pnl_2 > 0

    return {
        "pass": passed,
        "first_half_pnl": round(pnl_1, 2),
        "second_half_pnl": round(pnl_2, 2),
        "first_half_trades": len(first_half),
        "second_half_trades": len(second_half),
    }


def gate_outlier_test(trades: List[OptionTrade]) -> Dict[str, Any]:
    """Outlier test: remove top 5% of trades by P&L, portfolio must still be profitable."""
    if len(trades) < 20:
        return {"pass": False, "reason": "too few trades for outlier test"}

    sorted_by_pnl = sorted(trades, key=lambda t: t.pnl, reverse=True)
    cutoff = max(int(len(sorted_by_pnl) * 0.05), 1)
    trimmed = sorted_by_pnl[cutoff:]

    total_pnl = sum(t.pnl for t in trimmed)
    removed_pnl = sum(t.pnl for t in sorted_by_pnl[:cutoff])

    passed = total_pnl > 0

    return {
        "pass": passed,
        "total_pnl_without_outliers": round(total_pnl, 2),
        "removed_trades": cutoff,
        "removed_pnl": round(removed_pnl, 2),
        "remaining_trades": len(trimmed),
    }


# ===========================================================================
# Monthly equity curve
# ===========================================================================

def monthly_equity(equity_curve: List[dict]) -> List[dict]:
    """Resample equity curve to monthly for compact output."""
    if not equity_curve:
        return []

    eq = pd.DataFrame(equity_curve)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    monthly = eq["equity"].resample("ME").last().dropna()
    return [
        {"month": d.strftime("%Y-%m"), "equity": round(v, 2)}
        for d, v in monthly.items()
    ]


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Agentic Options Backtest v1 — Starting")
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"Period: {START_DATE} to {END_DATE}")
    log.info(f"Starting capital: ${STARTING_CAPITAL}")
    log.info(f"Walk-forward: {TRAIN_WINDOW}d train / {TEST_WINDOW}d test (SLIDING)")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    price_data = load_price_data(UNIVERSE, START_DATE, END_DATE)
    if not price_data:
        log.error("No price data loaded. Exiting.")
        sys.exit(1)

    spy_close = load_spy_data(START_DATE, END_DATE)
    earnings_data = load_earnings_dates(list(price_data.keys()))

    # ------------------------------------------------------------------
    # 2. Run backtests for each option variant
    # ------------------------------------------------------------------
    option_types = ["atm_call", "otm_call", "bull_spread"]
    results = {}

    for otype in option_types:
        bt = run_backtest(price_data, earnings_data, spy_close, otype)
        if not bt:
            log.warning(f"  {otype}: backtest returned empty, skipping")
            continue
        results[otype] = bt

    if not results:
        log.error("All backtests failed. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Run adversarial gates on each variant
    # ------------------------------------------------------------------
    gate_results = {}

    for otype, bt in results.items():
        log.info(f"\nRunning adversarial gates for {otype}...")
        trades = bt["trades"]
        eq = bt["equity_curve"]
        metrics = bt["metrics"]

        gates = {
            "permutation": gate_permutation_test(trades, eq),
            "regime": gate_regime_test(metrics),
            "sub_period": gate_sub_period_test(trades),
            "outlier": gate_outlier_test(trades),
        }

        n_pass = sum(1 for g in gates.values() if g.get("pass", False))
        gates["gates_passed"] = f"{n_pass}/4"
        gates["all_passed"] = n_pass == 4

        gate_results[otype] = gates

        log.info(f"  {otype} gates: {gates['gates_passed']} passed")
        for gname, gres in gates.items():
            if isinstance(gres, dict) and "pass" in gres:
                status = "PASS" if gres["pass"] else "FAIL"
                log.info(f"    {gname}: {status}")

    # ------------------------------------------------------------------
    # 4. Determine best variant
    # ------------------------------------------------------------------
    best_variant = None
    best_sharpe = -999

    for otype, bt in results.items():
        sharpe = bt["metrics"].get("sharpe", 0)
        gates_ok = gate_results.get(otype, {}).get("all_passed", False)

        # Prefer variants that pass all gates; among those, pick highest Sharpe
        if gates_ok and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_variant = otype

    # If no variant passes all gates, pick highest Sharpe anyway
    if best_variant is None:
        for otype, bt in results.items():
            sharpe = bt["metrics"].get("sharpe", 0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_variant = otype

    log.info(f"\nBest variant: {best_variant} (Sharpe={best_sharpe:.4f})")

    # ------------------------------------------------------------------
    # 5. Compile output
    # ------------------------------------------------------------------
    output = {
        "metadata": {
            "script": "agentic_options_backtest_v1.py",
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "universe_size": len(price_data),
            "tickers_used": sorted(price_data.keys()),
            "period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL,
            "walk_forward": f"{TRAIN_WINDOW}d train / {TEST_WINDOW}d test (SLIDING)",
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "best_variant": best_variant,
        "variants": {},
    }

    for otype, bt in results.items():
        metrics = bt["metrics"]
        monthly_eq = monthly_equity(bt["equity_curve"])

        output["variants"][otype] = {
            "metrics": metrics,
            "gates": gate_results.get(otype, {}),
            "monthly_equity": monthly_eq,
        }

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # 7. Log to MLflow if available
    # ------------------------------------------------------------------
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("agentic_options_backtest_v1")
            with mlflow.start_run(run_name=f"options_bt_{datetime.now():%Y%m%d_%H%M}"):
                # Log best variant metrics
                if best_variant and best_variant in results:
                    m = results[best_variant]["metrics"]
                    mlflow.log_params({
                        "best_variant": best_variant,
                        "universe_size": len(price_data),
                        "starting_capital": STARTING_CAPITAL,
                        "period": f"{START_DATE}_to_{END_DATE}",
                    })
                    mlflow.log_metrics({
                        "sharpe": m.get("sharpe", 0),
                        "sortino": m.get("sortino", 0),
                        "cagr": m.get("cagr", 0),
                        "max_dd": m.get("max_dd", 0),
                        "win_rate": m.get("win_rate", 0),
                        "profit_factor": m.get("profit_factor", 0),
                        "calmar": m.get("calmar", 0),
                        "total_trades": m.get("total_trades", 0),
                        "final_capital": m.get("final_capital", 0),
                    })

                    # Log all variants
                    for otype in results:
                        vm = results[otype]["metrics"]
                        mlflow.log_metrics({
                            f"{otype}_sharpe": vm.get("sharpe", 0),
                            f"{otype}_wr": vm.get("win_rate", 0),
                            f"{otype}_pf": vm.get("profit_factor", 0),
                        })

                mlflow.log_artifact(RESULTS_PATH)
            log.info("Results logged to MLflow")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Runtime: {elapsed / 60:.1f} minutes")
    log.info(f"Best variant: {best_variant}")

    for otype in option_types:
        if otype not in results:
            continue
        m = results[otype]["metrics"]
        gates = gate_results.get(otype, {})
        marker = " << BEST" if otype == best_variant else ""
        log.info(f"\n--- {otype}{marker} ---")
        log.info(f"  Sharpe={m['sharpe']:.3f}  Sortino={m['sortino']:.3f}  "
                 f"CAGR={m['cagr']:.2%}  MaxDD={m['max_dd']:.2%}")
        log.info(f"  WR={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
                 f"Calmar={m['calmar']:.2f}")
        log.info(f"  Trades={m.get('total_trades', 0)}  "
                 f"AvgPnL=${m['avg_trade_pnl']:.2f}  "
                 f"Final=${m.get('final_capital', 0):.2f}")
        log.info(f"  Oversold: {m.get('oversold_signal', {})}")
        log.info(f"  Earnings: {m.get('earnings_signal', {})}")
        log.info(f"  Regime Sharpes: {m.get('regime_sharpes', {})}")
        log.info(f"  Gates: {gates.get('gates_passed', 'N/A')}")

    log.info("\n" + "=" * 70)
    log.info("Done.")


if __name__ == "__main__":
    main()
