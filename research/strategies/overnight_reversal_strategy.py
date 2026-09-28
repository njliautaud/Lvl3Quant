#!/usr/bin/env python3
"""
Overnight Gap Reversal Strategy — Observation-Driven Mean Reversion
====================================================================

OBSERVATION (from market_observation_scanner.py):
  - 100% of S&P 500 stocks show positive overnight Sharpe ratio
  - Overnight gaps strongly REVERSE intraday (correlation = -0.96)
  - This is structural: institutional rebalancing, overnight risk premium unwind

HYPOTHESIS:
  When a stock gaps significantly overnight (> 1 standard deviation of its own
  trailing 63-day gap distribution), the intraday session will reverse a
  meaningful portion of that gap. We can trade the reversal.

STRATEGY:
  1. Universe: S&P 500 stocks, 5+ years daily data (yfinance)
  2. Signal: gap_pct = (Open - PrevClose) / PrevClose; threshold = 1 sigma
     of trailing 63-day gap distribution
  3. Entry: At the open, OPPOSITE direction of the gap
     (gap up -> short, gap down -> long)
  4. Exit: Close at MOC, or when 50% of the gap has reversed (whichever first)
  5. Sizing: Equal weight across signals, max 10 positions, prioritize by |gap|
  6. Filters: price >= $5, avg volume >= 500K (trailing 20d)

VALIDATION (HC #428 + HC #735):
  - Walk-forward: 60-day calibration, 1-day OOT, sliding window
  - Full OOT period (40+ days)
  - Regime stratification (green/red/flat using SPY close-to-close)
  - Permutation test (200 shuffles)
  - Lag sensitivity (T-1 signal)
  - Transaction costs: 5 bps each way (10 bps round-trip)
  - Reject if regime Sharpe gap > 0.50

Usage:
    python3 overnight_reversal_strategy.py [--no-cache] [--years 5] [--jobs 4]

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
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path("/home/jupiter/Lvl3Quant/data")
CACHE_PATH = CACHE_DIR / "overnight_reversal_cache.pkl"
OUTPUT_JSON = CACHE_DIR / "overnight_reversal_results.json"

# Strategy parameters
LOOKBACK_DAYS = 63          # Trailing window for gap sigma calculation
GAP_SIGMA_THRESHOLD = 1.0   # Min gap in sigmas to trigger signal
REVERSAL_TARGET = 0.50       # Exit when 50% of gap has reversed
MAX_POSITIONS = 10           # Max simultaneous positions per day
MIN_PRICE = 5.0              # Minimum stock price filter
MIN_AVG_VOLUME = 500_000     # Minimum 20-day average volume
COST_BPS_EACH_WAY = 5        # Transaction cost in basis points per leg
COST_BPS_RT = 2 * COST_BPS_EACH_WAY  # 10 bps round-trip

# Walk-forward parameters
WF_TRAIN_DAYS = 60           # Calibration window (sliding)
WF_TEST_DAYS = 1             # OOT window

# Validation parameters
N_PERMUTATIONS = 200         # Permutation test shuffles
MIN_OOT_DAYS = 40            # HC #428 minimum

# Regime classification thresholds for SPY daily returns
REGIME_FLAT_BPS = 25         # +/- 25 bps = flat day

# Wikipedia S&P 500 source
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# Data Acquisition
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        sym_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        tickers = df[sym_col].str.replace(".", "-", regex=False).tolist()
        print(f"  Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return sorted(set(tickers))
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using fallback list")
        # Minimal fallback — top 50 by weight
        return [
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMZN",
            "AVGO", "AXP", "BA", "BAC", "BLK", "BMY", "BRK-B", "C",
            "CAT", "CL", "CMCSA", "COP", "COST", "CRM", "CSCO", "CVX",
            "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "F", "FDX",
            "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON",
            "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN",
            "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET",
            "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
            "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
            "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T",
            "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
            "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
        ]


def download_data(tickers: list[str], years: int = 5, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for all tickers. Cache to disk."""
    import yfinance as yf

    if use_cache and CACHE_PATH.exists():
        age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"  Loading cached data ({age_hours:.1f}h old)")
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)

    end = dt.date.today()
    start = end - dt.timedelta(days=years * 365 + 30)
    print(f"  Downloading {len(tickers)} tickers from {start} to {end}")

    # Download SPY first (needed for regime classification)
    all_tickers = ["SPY"] + [t for t in tickers if t != "SPY"]
    data = {}
    batch_size = 50
    for i in tqdm(range(0, len(all_tickers), batch_size), desc="  Downloading"):
        batch = all_tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, start=str(start), end=str(end),
                              progress=False, threads=True, group_by="ticker")
            if len(batch) == 1:
                ticker = batch[0]
                if not raw.empty and len(raw) > LOOKBACK_DAYS + WF_TRAIN_DAYS:
                    data[ticker] = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for ticker in batch:
                    try:
                        df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                        df.dropna(subset=["Close", "Open", "Volume"], inplace=True)
                        if len(df) > LOOKBACK_DAYS + WF_TRAIN_DAYS:
                            data[ticker] = df
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"    Batch download failed: {e}")
        time.sleep(0.2)

    print(f"  Downloaded {len(data)} tickers with sufficient history")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute overnight gap features for every stock-day.

    Returns a DataFrame indexed by (date, ticker) with columns:
      - gap_pct: overnight gap as percentage
      - gap_sigma: gap normalized by trailing 63-day gap std
      - intraday_ret: (Close - Open) / Open (the reversal we want to capture)
      - close_ret: close-to-close return
      - prev_close: prior day's close price
      - open_price: today's open
      - close_price: today's close
      - high_price: today's high
      - low_price: today's low
      - avg_volume_20d: trailing 20-day average volume
      - price_ok: meets minimum price filter
      - volume_ok: meets minimum volume filter
    """
    records = []

    for ticker, df in tqdm(data.items(), desc="  Computing features"):
        if ticker == "SPY":
            continue  # SPY is used for regime, not as a trading candidate

        df = df.copy().sort_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Overnight gap
        prev_close = df["Close"].shift(1)
        gap_pct = (df["Open"] - prev_close) / prev_close

        # Trailing 63-day gap standard deviation (for sigma normalization)
        gap_std = gap_pct.rolling(LOOKBACK_DAYS, min_periods=30).std()
        gap_mean = gap_pct.rolling(LOOKBACK_DAYS, min_periods=30).mean()
        gap_sigma = (gap_pct - gap_mean) / gap_std

        # Intraday return (what we're trying to capture)
        intraday_ret = (df["Close"] - df["Open"]) / df["Open"]

        # Volume filter
        avg_vol_20 = df["Volume"].rolling(20, min_periods=10).mean()

        for i in range(LOOKBACK_DAYS + 1, len(df)):
            date = df.index[i]
            if pd.isna(gap_pct.iloc[i]) or pd.isna(gap_std.iloc[i]):
                continue
            if gap_std.iloc[i] < 1e-8:
                continue

            records.append({
                "date": date,
                "ticker": ticker,
                "gap_pct": gap_pct.iloc[i],
                "gap_sigma": gap_sigma.iloc[i],
                "intraday_ret": intraday_ret.iloc[i],
                "close_ret": (df["Close"].iloc[i] / df["Close"].iloc[i - 1]) - 1,
                "prev_close": prev_close.iloc[i],
                "open_price": df["Open"].iloc[i],
                "close_price": df["Close"].iloc[i],
                "high_price": df["High"].iloc[i],
                "low_price": df["Low"].iloc[i],
                "avg_volume_20d": avg_vol_20.iloc[i],
                "price_ok": df["Close"].iloc[i - 1] >= MIN_PRICE,
                "volume_ok": avg_vol_20.iloc[i] >= MIN_AVG_VOLUME if not pd.isna(avg_vol_20.iloc[i]) else False,
            })

    features = pd.DataFrame(records)
    if features.empty:
        raise ValueError("No features computed — check data download")

    features["date"] = pd.to_datetime(features["date"])
    features.set_index(["date", "ticker"], inplace=True)
    features.sort_index(inplace=True)
    print(f"  Feature matrix: {len(features):,} stock-days, "
          f"{features.index.get_level_values('date').nunique()} trading days, "
          f"{features.index.get_level_values('ticker').nunique()} tickers")
    return features


def compute_spy_regime(data: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Classify each trading day as green/red/flat using SPY close-to-close.

    Returns a Series indexed by date with values in {'green', 'red', 'flat'}.
    """
    spy = data.get("SPY")
    if spy is None:
        raise ValueError("SPY data required for regime classification")

    spy = spy.copy().sort_index()
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy_ret = spy["Close"].pct_change()
    threshold = REGIME_FLAT_BPS / 10000

    def classify(r):
        if pd.isna(r):
            return "flat"
        if r > threshold:
            return "green"
        if r < -threshold:
            return "red"
        return "flat"

    regime = spy_ret.apply(classify)
    regime.name = "regime"
    return regime


# ---------------------------------------------------------------------------
# Strategy Simulation
# ---------------------------------------------------------------------------

def simulate_day(day_features: pd.DataFrame, sigma_threshold: float = GAP_SIGMA_THRESHOLD) -> dict:
    """
    Simulate one day of the overnight reversal strategy.

    Args:
        day_features: DataFrame of all stock features for one day
        sigma_threshold: minimum |gap_sigma| to trigger a signal

    Returns:
        dict with keys: date, n_signals, n_trades, returns (list), gross_return, net_return
    """
    # Apply filters
    eligible = day_features[
        (day_features["price_ok"]) &
        (day_features["volume_ok"]) &
        (day_features["gap_sigma"].abs() >= sigma_threshold)
    ].copy()

    if eligible.empty:
        return {"n_signals": 0, "n_trades": 0, "returns": [], "gross_return": 0.0, "net_return": 0.0}

    # Sort by absolute gap magnitude (strongest gaps first)
    eligible = eligible.sort_values("gap_sigma", key=lambda x: x.abs(), ascending=False)

    # Take top MAX_POSITIONS
    trades = eligible.head(MAX_POSITIONS)

    trade_returns = []
    for _, row in trades.iterrows():
        gap_pct = row["gap_pct"]
        open_price = row["open_price"]
        close_price = row["close_price"]
        high_price = row["high_price"]
        low_price = row["low_price"]

        # Direction: OPPOSITE of gap
        # Gap up -> short (we expect price to come down)
        # Gap down -> long (we expect price to come up)
        direction = -1 if gap_pct > 0 else 1  # -1 = short, +1 = long

        # Check if 50% reversal target was hit intraday
        # For a gap up (short trade): target = open - 0.5 * gap_in_dollars
        # For a gap down (long trade): target = open + 0.5 * |gap_in_dollars|
        gap_dollars = gap_pct * row["prev_close"]
        target_price = open_price - direction * abs(gap_dollars) * REVERSAL_TARGET * (-1)
        # Simplify: if shorting (gap up), target is BELOW open
        #           if long (gap down), target is ABOVE open

        if direction == -1:
            # Short trade: target = open - 0.5 * gap_dollars (gap_dollars > 0)
            target_price = open_price - REVERSAL_TARGET * abs(gap_dollars)
            # Did price reach target? Low must be <= target
            if low_price <= target_price:
                # Hit target — return is (open - target) / open
                trade_ret = (open_price - target_price) / open_price
            else:
                # Held to close — return is (open - close) / open
                trade_ret = (open_price - close_price) / open_price
        else:
            # Long trade: target = open + 0.5 * |gap_dollars| (gap_dollars < 0)
            target_price = open_price + REVERSAL_TARGET * abs(gap_dollars)
            # Did price reach target? High must be >= target
            if high_price >= target_price:
                trade_ret = (target_price - open_price) / open_price
            else:
                trade_ret = (close_price - open_price) / open_price

        trade_returns.append(trade_ret)

    n_trades = len(trade_returns)
    # Equal weight across trades
    gross_return = np.mean(trade_returns) if trade_returns else 0.0
    # Transaction costs: 10 bps round-trip on each trade
    cost = COST_BPS_RT / 10000
    net_return = gross_return - cost

    return {
        "n_signals": len(eligible),
        "n_trades": n_trades,
        "returns": trade_returns,
        "gross_return": gross_return,
        "net_return": net_return,
    }


def run_backtest(features: pd.DataFrame, regime: pd.Series,
                 sigma_threshold: float = GAP_SIGMA_THRESHOLD,
                 start_date=None, end_date=None) -> pd.DataFrame:
    """
    Run the full backtest across all dates.

    Returns a DataFrame with one row per trading day.
    """
    dates = features.index.get_level_values("date").unique().sort_values()
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]

    results = []
    for date in dates:
        try:
            day_data = features.loc[date]
        except KeyError:
            continue

        if isinstance(day_data, pd.Series):
            # Only one stock on this day
            day_data = day_data.to_frame().T

        sim = simulate_day(day_data, sigma_threshold)
        regime_label = regime.get(date, "flat") if date in regime.index else "flat"

        results.append({
            "date": date,
            "n_signals": sim["n_signals"],
            "n_trades": sim["n_trades"],
            "gross_return": sim["gross_return"],
            "net_return": sim["net_return"],
            "regime": regime_label,
        })

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_validation(features: pd.DataFrame, regime: pd.Series) -> dict:
    """
    Walk-forward validation with 60-day calibration, 1-day OOT, sliding window.

    The calibration window is used to verify the strategy 'works' in-sample
    before trusting the OOT day. This mimics real deployment where we'd
    only trade if recent history supports the strategy.

    Returns dict with IS and OOT results.
    """
    dates = features.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)

    if n_dates < WF_TRAIN_DAYS + MIN_OOT_DAYS:
        raise ValueError(f"Not enough dates for walk-forward: {n_dates} < {WF_TRAIN_DAYS + MIN_OOT_DAYS}")

    oot_results = []
    is_results = []

    for i in tqdm(range(WF_TRAIN_DAYS, n_dates), desc="  Walk-forward"):
        train_start = dates[i - WF_TRAIN_DAYS]
        train_end = dates[i - 1]
        test_date = dates[i]

        # In-sample: run backtest on calibration window
        train_bt = run_backtest(features, regime, start_date=train_start, end_date=train_end)
        if train_bt.empty or train_bt["n_trades"].sum() == 0:
            continue

        # Check if IS strategy was profitable (basic go/no-go gate)
        is_sharpe = compute_sharpe(train_bt["net_return"])
        is_mean = train_bt["net_return"].mean()

        is_results.append({
            "date": test_date,
            "is_sharpe": is_sharpe,
            "is_mean": is_mean,
            "is_days": len(train_bt),
            "is_trades": int(train_bt["n_trades"].sum()),
        })

        # Only trade OOT if IS Sharpe > 0 (basic quality gate)
        if is_sharpe <= 0:
            oot_results.append({
                "date": test_date,
                "traded": False,
                "net_return": 0.0,
                "gross_return": 0.0,
                "n_trades": 0,
                "regime": regime.get(test_date, "flat") if test_date in regime.index else "flat",
                "is_sharpe": is_sharpe,
            })
            continue

        # OOT: simulate one day
        try:
            day_data = features.loc[test_date]
        except KeyError:
            continue

        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T

        sim = simulate_day(day_data)
        regime_label = regime.get(test_date, "flat") if test_date in regime.index else "flat"

        oot_results.append({
            "date": test_date,
            "traded": True,
            "net_return": sim["net_return"],
            "gross_return": sim["gross_return"],
            "n_trades": sim["n_trades"],
            "regime": regime_label,
            "is_sharpe": is_sharpe,
        })

    oot_df = pd.DataFrame(oot_results)
    is_df = pd.DataFrame(is_results)
    oot_df["date"] = pd.to_datetime(oot_df["date"])
    is_df["date"] = pd.to_datetime(is_df["date"])
    oot_df.set_index("date", inplace=True)
    is_df.set_index("date", inplace=True)

    return {"oot": oot_df, "is": is_df}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """Annualized Sharpe ratio (assume 252 trading days)."""
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(252)
    return float(sr)


def compute_sortino(returns: pd.Series, annualize: bool = True) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 1 or downside.std() < 1e-10:
        return float("inf") if returns.mean() > 0 else 0.0
    sr = returns.mean() / downside.std()
    if annualize:
        sr *= np.sqrt(252)
    return float(sr)


def compute_profit_factor(returns: pd.Series) -> float:
    """Profit factor = sum(wins) / |sum(losses)|."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def compute_win_rate(returns: pd.Series) -> float:
    """Fraction of positive return days."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def compute_cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    if len(returns) < 2:
        return 0.0
    cum = (1 + returns).prod()
    years = len(returns) / 252
    if years < 0.01 or cum <= 0:
        return 0.0
    return float(cum ** (1 / years) - 1)


def compute_full_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute all strategy metrics."""
    if len(returns) == 0:
        return {"label": label, "n_days": 0}
    return {
        "label": label,
        "n_days": int(len(returns)),
        "sharpe": round(compute_sharpe(returns), 3),
        "sortino": round(compute_sortino(returns), 3),
        "profit_factor": round(compute_profit_factor(returns), 3),
        "win_rate": round(compute_win_rate(returns), 4),
        "max_drawdown": round(compute_max_drawdown(returns), 4),
        "cagr": round(compute_cagr(returns), 4),
        "mean_daily_ret_bps": round(returns.mean() * 10000, 2),
        "std_daily_ret_bps": round(returns.std() * 10000, 2),
        "total_return_pct": round(((1 + returns).prod() - 1) * 100, 2),
        "skew": round(float(returns.skew()), 3),
        "kurtosis": round(float(returns.kurtosis()), 3),
    }


# ---------------------------------------------------------------------------
# Regime Stratification
# ---------------------------------------------------------------------------

def regime_analysis(results: pd.DataFrame) -> dict:
    """
    Stratify performance by market regime (green/red/flat days).

    HC #428: Reject if |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) > 0.50
    """
    analysis = {}
    for regime in ["green", "red", "flat"]:
        mask = results["regime"] == regime
        rets = results.loc[mask, "net_return"]
        if len(rets) > 0:
            analysis[regime] = compute_full_metrics(rets, label=f"regime_{regime}")
        else:
            analysis[regime] = {"label": f"regime_{regime}", "n_days": 0, "sharpe": 0.0}

    # Regime gap test (HC #428)
    sharpe_green = analysis.get("green", {}).get("sharpe", 0.0)
    sharpe_red = analysis.get("red", {}).get("sharpe", 0.0)
    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    if max_abs > 0:
        regime_gap = abs(sharpe_green - sharpe_red) / max_abs
    else:
        regime_gap = 0.0

    analysis["regime_gap"] = round(regime_gap, 4)
    analysis["regime_gap_pass"] = regime_gap <= 0.50

    return analysis


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(results: pd.DataFrame, n_perms: int = N_PERMUTATIONS) -> dict:
    """
    Permutation test: shuffle daily returns, recompute Sharpe.
    p-value = fraction of permuted Sharpes >= observed Sharpe.
    """
    returns = results["net_return"].values
    observed_sharpe = compute_sharpe(pd.Series(returns))

    perm_sharpes = []
    rng = np.random.RandomState(42)
    for _ in tqdm(range(n_perms), desc="  Permutation test"):
        shuffled = rng.permutation(returns)
        perm_sharpes.append(compute_sharpe(pd.Series(shuffled)))

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= observed_sharpe))

    return {
        "observed_sharpe": round(observed_sharpe, 4),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 4),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perms,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


# ---------------------------------------------------------------------------
# Lag Sensitivity
# ---------------------------------------------------------------------------

def lag_sensitivity(features: pd.DataFrame, regime: pd.Series) -> dict:
    """
    Test if T-1 signal (yesterday's gap) still predicts today's reversal.
    This checks signal persistence / look-ahead bias.
    """
    # Create lagged features: shift gap_sigma by 1 day per ticker
    dates = features.index.get_level_values("date").unique().sort_values()
    tickers = features.index.get_level_values("ticker").unique()

    lagged_records = []
    for ticker in tickers:
        try:
            ticker_data = features.xs(ticker, level="ticker").copy()
        except KeyError:
            continue
        if len(ticker_data) < 2:
            continue

        # Shift gap_sigma by 1 day
        ticker_data["gap_sigma_lag1"] = ticker_data["gap_sigma"].shift(1)
        for date in ticker_data.index[1:]:
            row = ticker_data.loc[date]
            if pd.isna(row["gap_sigma_lag1"]):
                continue
            lagged_records.append({
                "date": date,
                "ticker": ticker,
                "gap_sigma": row["gap_sigma_lag1"],  # Use LAGGED signal
                "gap_pct": row.get("gap_pct", 0),
                "intraday_ret": row["intraday_ret"],
                "prev_close": row["prev_close"],
                "open_price": row["open_price"],
                "close_price": row["close_price"],
                "high_price": row["high_price"],
                "low_price": row["low_price"],
                "avg_volume_20d": row["avg_volume_20d"],
                "price_ok": row["price_ok"],
                "volume_ok": row["volume_ok"],
            })

    if not lagged_records:
        return {"lag1_sharpe": 0.0, "lag1_days": 0, "signal_decays": True}

    lagged_df = pd.DataFrame(lagged_records)
    lagged_df["date"] = pd.to_datetime(lagged_df["date"])
    lagged_df.set_index(["date", "ticker"], inplace=True)

    lag_bt = run_backtest(lagged_df, regime)
    lag_metrics = compute_full_metrics(lag_bt["net_return"], label="lag1")

    return {
        "lag1_sharpe": lag_metrics.get("sharpe", 0.0),
        "lag1_win_rate": lag_metrics.get("win_rate", 0.0),
        "lag1_days": lag_metrics.get("n_days", 0),
        "signal_decays": lag_metrics.get("sharpe", 0.0) < compute_sharpe(
            run_backtest(features, regime)["net_return"]
        ) * 0.5,
        "note": "T-1 signal should be much weaker — confirms signal is timely, not persistent bias",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_metrics(metrics: dict, indent: int = 2):
    """Print metrics dict in a readable format."""
    prefix = " " * indent
    for k, v in metrics.items():
        if k == "label":
            continue
        if isinstance(v, float):
            print(f"{prefix}{k:>25s}: {v:>12.4f}")
        elif isinstance(v, int):
            print(f"{prefix}{k:>25s}: {v:>12d}")
        elif isinstance(v, bool):
            print(f"{prefix}{k:>25s}: {'PASS' if v else 'FAIL':>12s}")
        else:
            print(f"{prefix}{k:>25s}: {str(v):>12s}")


def generate_report(full_bt: pd.DataFrame, wf_results: dict,
                    regime_results: dict, perm_results: dict,
                    lag_results: dict) -> dict:
    """Generate the full validation report."""

    print_section("OVERNIGHT GAP REVERSAL STRATEGY — VALIDATION REPORT")

    # --- Observation chain ---
    print("\n  OBSERVATION -> HYPOTHESIS -> STRATEGY -> VALIDATION")
    print("  " + "-" * 50)
    print("  Observation: 100% of S&P 500 stocks show positive overnight Sharpe;")
    print("               overnight gaps reverse intraday (correlation = -0.96)")
    print("  Hypothesis:  Significant gaps (>1 sigma) reverse meaningfully intraday")
    print("  Strategy:    Trade opposite direction of gap, exit at MOC or 50% reversal")
    print("  Validation:  Walk-forward, regime-stratified, permutation-tested")

    # --- Full-sample backtest ---
    print_section("FULL-SAMPLE BACKTEST (GROSS & NET)")
    gross_metrics = compute_full_metrics(full_bt["gross_return"], "gross")
    net_metrics = compute_full_metrics(full_bt["net_return"], "net")
    print("\n  GROSS (before costs):")
    print_metrics(gross_metrics)
    print("\n  NET (after 10 bps RT costs):")
    print_metrics(net_metrics)

    traded_days = full_bt[full_bt["n_trades"] > 0]
    print(f"\n  Trading days: {len(traded_days)} / {len(full_bt)} ({100*len(traded_days)/max(len(full_bt),1):.1f}%)")
    print(f"  Avg trades/day (when trading): {traded_days['n_trades'].mean():.1f}")
    print(f"  Avg signals/day (when trading): {traded_days['n_signals'].mean():.1f}")

    # --- Walk-forward OOT ---
    print_section("WALK-FORWARD OOT RESULTS")
    oot_df = wf_results["oot"]
    traded_oot = oot_df[oot_df["traded"]]
    if len(traded_oot) > 0:
        oot_metrics = compute_full_metrics(traded_oot["net_return"], "oot_traded")
        print_metrics(oot_metrics)
        print(f"\n  OOT days traded: {len(traded_oot)}")
        print(f"  OOT days skipped (IS gate): {len(oot_df) - len(traded_oot)}")
        oot_pass = len(traded_oot) >= MIN_OOT_DAYS
        print(f"  Meets HC #428 minimum ({MIN_OOT_DAYS}d): {'PASS' if oot_pass else 'FAIL'}")
    else:
        oot_metrics = {"sharpe": 0.0, "n_days": 0}
        oot_pass = False
        print("  No OOT days traded — strategy never passed IS gate")

    # --- Regime stratification ---
    print_section("REGIME STRATIFICATION")
    for regime in ["green", "red", "flat"]:
        r = regime_results.get(regime, {})
        print(f"\n  {regime.upper()} days ({r.get('n_days', 0)} days):")
        if r.get("n_days", 0) > 0:
            print_metrics(r)
    print(f"\n  Regime gap (HC #428): {regime_results['regime_gap']:.4f} "
          f"({'PASS' if regime_results['regime_gap_pass'] else 'FAIL'} threshold 0.50)")

    # --- Permutation test ---
    print_section("PERMUTATION TEST")
    print_metrics(perm_results)

    # --- Lag sensitivity ---
    print_section("LAG SENSITIVITY (T-1 SIGNAL)")
    print_metrics(lag_results)

    # --- Final verdict ---
    print_section("FINAL VERDICT")
    checks = {
        "OOT Sharpe > 0": oot_metrics.get("sharpe", 0) > 0,
        f"OOT days >= {MIN_OOT_DAYS}": oot_pass,
        "Regime gap <= 0.50": regime_results["regime_gap_pass"],
        "Permutation p < 0.05": perm_results["significant_5pct"],
        "Signal decays with lag": lag_results.get("signal_decays", False),
    }
    all_pass = all(checks.values())
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    print(f"\n  OVERALL: {'ACCEPTED' if all_pass else 'NEEDS REVIEW'}")

    # Build output dict
    output = {
        "strategy": "overnight_gap_reversal",
        "observation": "100% positive overnight Sharpe, gap-intraday correlation = -0.96",
        "hypothesis": "Significant overnight gaps (>1 sigma) reverse intraday",
        "parameters": {
            "gap_sigma_threshold": GAP_SIGMA_THRESHOLD,
            "reversal_target": REVERSAL_TARGET,
            "max_positions": MAX_POSITIONS,
            "lookback_days": LOOKBACK_DAYS,
            "min_price": MIN_PRICE,
            "min_avg_volume": MIN_AVG_VOLUME,
            "cost_bps_rt": COST_BPS_RT,
        },
        "full_sample": {
            "gross": gross_metrics,
            "net": net_metrics,
            "trading_days": int(len(traded_days)),
            "total_days": int(len(full_bt)),
        },
        "walk_forward_oot": oot_metrics,
        "regime_analysis": {
            k: v for k, v in regime_results.items()
            if not isinstance(v, pd.DataFrame)
        },
        "permutation_test": perm_results,
        "lag_sensitivity": lag_results,
        "validation_checks": {k: v for k, v in checks.items()},
        "verdict": "ACCEPTED" if all_pass else "NEEDS_REVIEW",
        "generated_at": dt.datetime.now().isoformat(),
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overnight Gap Reversal Strategy")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download data")
    parser.add_argument("--years", type=int, default=5, help="Years of history to download")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel download threads (unused, yfinance handles)")
    args = parser.parse_args()

    print("=" * 70)
    print("  OVERNIGHT GAP REVERSAL STRATEGY")
    print("  Observation-driven mean reversion on S&P 500")
    print("=" * 70)

    # Step 1: Get universe
    print("\n[1/7] Fetching S&P 500 universe...")
    tickers = get_sp500_tickers()

    # Step 2: Download data
    print("\n[2/7] Downloading daily OHLCV data...")
    data = download_data(tickers, years=args.years, use_cache=not args.no_cache)

    # Step 3: Compute features
    print("\n[3/7] Computing overnight gap features...")
    features = compute_features(data)
    regime = compute_spy_regime(data)

    # Step 4: Full-sample backtest
    print("\n[4/7] Running full-sample backtest...")
    full_bt = run_backtest(features, regime)
    print(f"  Full backtest: {len(full_bt)} days, "
          f"{full_bt['n_trades'].sum():.0f} total trades")

    # Step 5: Walk-forward validation
    print("\n[5/7] Running walk-forward validation (60d train, 1d OOT)...")
    wf_results = walk_forward_validation(features, regime)

    # Step 6: Regime analysis (on OOT results)
    print("\n[6/7] Regime stratification & permutation test...")
    oot_traded = wf_results["oot"][wf_results["oot"]["traded"]]
    if len(oot_traded) > 0:
        regime_results = regime_analysis(oot_traded)
        perm_results = permutation_test(oot_traded)
    else:
        # Fallback to full backtest if no OOT trades
        regime_results = regime_analysis(full_bt)
        perm_results = permutation_test(full_bt)

    # Step 7: Lag sensitivity
    print("\n[7/7] Lag sensitivity analysis...")
    lag_results = lag_sensitivity(features, regime)

    # Generate report
    output = generate_report(full_bt, wf_results, regime_results, perm_results, lag_results)

    # Save results
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    output = make_serializable(output)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_JSON}")
    print(f"  Done. Total time: script execution complete.")

    return output


if __name__ == "__main__":
    main()
