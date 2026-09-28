#!/usr/bin/env python3
"""
Institutional Momentum / 13F Following Strategy Backtester
==========================================================
Academic basis: Stocks where hedge funds are building positions (increasing
ownership) tend to outperform over the next quarter. Well-documented anomaly
(Brunnermeier & Nagel 2004; Yan & Zhang 2009; Chen, Jegadeesh, Wermers 2000).

Proxy approach (no real-time 13F data):
  1. Volume-Price Divergence: Rising volume + stable/rising price = accumulation
  2. OBV Breakout: OBV at new 20-day high before price hits new high
  3. Relative Volume: Sustained 1.5x+ avg volume for 5+ consecutive days
  4. Price Consolidation + Volume: Tight range (<3%) over 10 days + above-avg vol

Entry: Buy when 2+ (or 3+ for strict) of the 4 signals trigger
Exit: Hold for 40 days (or 20d for short-hold variant)
Regime hedge: Half-size when SPY < 200-SMA
Universe: S&P 500 stocks
Cost: $0 commission (RH), 0.02% slippage

Variants tested:
  A: Base (2+ signals, 40d hold)
  B: Strict (3+ signals, 40d hold)
  C: With earnings filter (within 10 days of earnings)
  D: With momentum filter (RSI > 50)
  E: Short hold (20d)

5-Gate validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (100 shuffles)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

OOT: Jan 2022 - Jul 2026
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

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/institutional_momentum")
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/institutional_momentum_results.json")

# Dates
DATA_START = "2021-01-01"   # Need lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"

# Cost model
SLIPPAGE_BPS = 2  # 0.02% = 2 bps
COMMISSION = 0.0

# Position sizing
STARTING_CAPITAL = 10000.0
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.10  # 10% per position

# Signal parameters
OBV_LOOKBACK = 20
VOLUME_AVG_PERIOD = 20
CONSEC_HIGH_VOL_DAYS = 5
HIGH_VOL_THRESHOLD = 1.5
CONSOLIDATION_DAYS = 10
CONSOLIDATION_RANGE_PCT = 0.03  # 3%
RSI_PERIOD = 14

# Number of S&P500 stocks to sample (full universe is slow with yfinance)
# We use a representative sample of ~100 liquid stocks
SAMPLE_SIZE = 100
PERMUTATION_ITERS = 100

# ─── S&P 500 representative sample ──────────────────────────────────────────

SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "MA", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "LLY", "AVGO", "WMT", "MCD", "TMO",
    "CSCO", "ACN", "DHR", "ABT", "NEE", "LIN", "TXN", "PM", "UNP",
    "RTX", "AMGN", "HON", "LOW", "IBM", "QCOM", "SPGI", "GE", "CAT",
    "BA", "SBUX", "MDT", "INTU", "BLK", "DE", "GILD", "ADI", "ISRG",
    "AXP", "MMC", "SYK", "PLD", "VRTX", "MDLZ", "TJX", "CI", "CB",
    "REGN", "ZTS", "MO", "CL", "DUK", "SO", "CME", "ICE", "PNC",
    "USB", "EOG", "SLB", "APD", "EMR", "NSC", "WM", "GD", "FDX",
    "MCK", "CCI", "PSA", "ORLY", "AEP", "D", "SRE", "KMB", "F",
    "GM", "DAL", "LUV", "AAL", "CCL", "NCLH", "PARA", "WBA", "VZ",
    "T", "INTC", "PYPL",
]


# ─── Data download ──────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for tickers, with caching."""
    data = {}
    to_download = []

    for t in tickers:
        cache_file = CACHE_DIR / f"{t}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            if len(df) > 50:
                data[t] = df
                continue
        to_download.append(t)

    if to_download:
        log.info(f"Downloading {len(to_download)} tickers from yfinance...")
        # Download in batches to avoid timeouts
        batch_size = 20
        for i in range(0, len(to_download), batch_size):
            batch = to_download[i:i + batch_size]
            log.info(f"  Batch {i // batch_size + 1}: {len(batch)} tickers")
            try:
                raw = yf.download(
                    batch, start=start, end=end,
                    group_by="ticker", auto_adjust=True,
                    threads=True, progress=False,
                )
                if len(batch) == 1:
                    t = batch[0]
                    if len(raw) > 50:
                        raw.to_parquet(CACHE_DIR / f"{t}.parquet")
                        data[t] = raw
                else:
                    for t in batch:
                        try:
                            df = raw[t].dropna(how="all")
                            if len(df) > 50:
                                df.to_parquet(CACHE_DIR / f"{t}.parquet")
                                data[t] = df
                        except Exception:
                            pass
            except Exception as e:
                log.warning(f"  Batch download failed: {e}")

    # Download SPY for regime filter
    spy_cache = CACHE_DIR / "SPY.parquet"
    if spy_cache.exists():
        data["SPY"] = pd.read_parquet(spy_cache)
    else:
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if len(spy) > 50:
            spy.to_parquet(spy_cache)
            data["SPY"] = spy

    log.info(f"Total tickers with data: {len(data) - 1} + SPY")
    return data


# ─── Indicator calculations ─────────────────────────────────────────────────

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 4 institutional accumulation proxy signals for a stock."""
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["volume"] = volume

    # Volume moving average
    vol_ma = volume.rolling(VOLUME_AVG_PERIOD).mean()
    out["vol_ratio"] = volume / vol_ma.replace(0, np.nan)

    # OBV
    obv = compute_obv(close, volume)
    out["obv"] = obv
    out["obv_20high"] = obv.rolling(OBV_LOOKBACK).max()
    out["price_20high"] = close.rolling(OBV_LOOKBACK).max()

    # RSI
    out["rsi"] = compute_rsi(close, RSI_PERIOD)

    # --- Signal 1: Volume-Price Divergence ---
    # Rising volume (5-day vol avg > 20-day vol avg) with stable/rising price (5d return >= 0)
    vol_5d = volume.rolling(5).mean()
    price_ret_5d = close.pct_change(5)
    out["sig_vol_price_div"] = ((vol_5d > vol_ma * 1.2) & (price_ret_5d >= 0)).astype(int)

    # --- Signal 2: OBV Breakout ---
    # OBV at new 20-day high while price is NOT at 20-day high
    obv_at_high = (obv >= out["obv_20high"] * 0.999)  # within 0.1% of high
    price_below_high = (close < out["price_20high"] * 0.98)  # at least 2% below high
    out["sig_obv_breakout"] = (obv_at_high & price_below_high).astype(int)

    # --- Signal 3: Sustained High Relative Volume ---
    # Volume > 1.5x average for 5+ consecutive days
    high_vol = (out["vol_ratio"] > HIGH_VOL_THRESHOLD).astype(int)
    # Count consecutive high-vol days
    consec = high_vol.copy()
    for i in range(1, CONSEC_HIGH_VOL_DAYS):
        consec = consec & high_vol.shift(i).fillna(0).astype(int)
    out["sig_sustained_vol"] = consec

    # --- Signal 4: Price Consolidation + Volume ---
    # Tight range (<3%) over 10 days with above-average volume
    rolling_high = close.rolling(CONSOLIDATION_DAYS).max()
    rolling_low = close.rolling(CONSOLIDATION_DAYS).min()
    range_pct = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)
    vol_above_avg = (out["vol_ratio"] > 1.1)  # at least 10% above average
    out["sig_consolidation_vol"] = ((range_pct < CONSOLIDATION_RANGE_PCT) & vol_above_avg).astype(int)

    # Total signal count
    out["signal_count"] = (
        out["sig_vol_price_div"]
        + out["sig_obv_breakout"]
        + out["sig_sustained_vol"]
        + out["sig_consolidation_vol"]
    )

    return out


# ─── Earnings dates (for variant C) ─────────────────────────────────────────

def get_earnings_dates(ticker: str) -> set:
    """Get approximate earnings dates for a ticker."""
    try:
        t = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=50)
        if cal is not None and len(cal) > 0:
            return set(cal.index.tz_localize(None).normalize())
    except Exception:
        pass
    return set()


# ─── Backtest engine ────────────────────────────────────────────────────────

def run_backtest(
    all_signals: dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
    min_signals: int = 2,
    hold_days: int = 40,
    require_rsi_above_50: bool = False,
    earnings_filter: bool = False,
    earnings_dates: dict[str, set] | None = None,
    label: str = "base",
) -> dict:
    """Run the institutional momentum backtest."""
    log.info(f"Running variant '{label}': min_signals={min_signals}, hold={hold_days}d, "
             f"rsi_filter={require_rsi_above_50}, earnings_filter={earnings_filter}")

    # Compute SPY 200-SMA for regime filter
    spy_close = spy_data["Close"].squeeze()
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_sma200 = spy_close.rolling(200).mean()

    # Build universe of dates
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    trades = []
    portfolio_value = [STARTING_CAPITAL]
    portfolio_dates = [oot_start]
    active_positions = []  # list of dicts: {ticker, entry_date, entry_price, shares, exit_date_target, half_size}

    # Get all trading dates from SPY
    trading_dates = spy_close.loc[oot_start:oot_end].index

    capital = STARTING_CAPITAL

    for date in trading_dates:
        # Check for exits
        new_active = []
        for pos in active_positions:
            if date >= pos["exit_date_target"]:
                # Exit
                ticker = pos["ticker"]
                if ticker in all_signals and date in all_signals[ticker].index:
                    exit_price = all_signals[ticker].loc[date, "close"]
                elif ticker in all_signals:
                    # Find nearest available date
                    avail = all_signals[ticker].index[all_signals[ticker].index >= date]
                    if len(avail) > 0:
                        exit_price = all_signals[ticker].loc[avail[0], "close"]
                    else:
                        exit_price = pos["entry_price"]  # fallback
                else:
                    exit_price = pos["entry_price"]

                if pd.isna(exit_price) or exit_price <= 0:
                    new_active.append(pos)
                    continue

                slippage = exit_price * SLIPPAGE_BPS / 10000
                net_exit = exit_price - slippage
                pnl = (net_exit - pos["entry_price"]) * pos["shares"]
                ret = (net_exit / pos["entry_price"]) - 1.0
                capital += pos["shares"] * net_exit
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(net_exit, 2),
                    "shares": round(pos["shares"], 4),
                    "pnl": round(pnl, 2),
                    "return": round(ret, 4),
                    "half_size": pos["half_size"],
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check for new entries (only if we have capacity)
        if len(active_positions) < MAX_POSITIONS:
            # Regime check
            bear_market = False
            if date in spy_sma200.index and not pd.isna(spy_sma200.loc[date]):
                bear_market = spy_close.loc[date] < spy_sma200.loc[date]

            candidates = []
            for ticker, sig_df in all_signals.items():
                if ticker == "SPY":
                    continue
                if date not in sig_df.index:
                    continue

                row = sig_df.loc[date]
                if row["signal_count"] < min_signals:
                    continue

                if require_rsi_above_50 and (pd.isna(row["rsi"]) or row["rsi"] <= 50):
                    continue

                if earnings_filter and earnings_dates is not None:
                    # Only trade if within 10 days of earnings
                    edates = earnings_dates.get(ticker, set())
                    if not edates:
                        continue
                    near_earnings = any(
                        abs((date - ed).days) <= 10 for ed in edates
                    )
                    if not near_earnings:
                        continue

                # Skip if already in position
                if any(p["ticker"] == ticker for p in active_positions):
                    continue

                candidates.append((ticker, row["signal_count"], row["close"]))

            # Sort by signal strength (more signals = better)
            candidates.sort(key=lambda x: -x[1])

            # Enter positions
            slots = MAX_POSITIONS - len(active_positions)
            for ticker, sig_count, price in candidates[:slots]:
                if pd.isna(price) or price <= 0:
                    continue
                size_frac = POSITION_SIZE_PCT * (0.5 if bear_market else 1.0)
                alloc = capital * size_frac
                slippage = price * SLIPPAGE_BPS / 10000
                net_entry = price + slippage
                shares = alloc / net_entry
                if shares * net_entry < 5:  # minimum $5 trade
                    continue

                capital -= shares * net_entry

                # Find target exit date
                future_dates = trading_dates[trading_dates > date]
                if len(future_dates) >= hold_days:
                    exit_target = future_dates[hold_days - 1]
                elif len(future_dates) > 0:
                    exit_target = future_dates[-1]
                else:
                    continue

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "entry_price": net_entry,
                    "shares": shares,
                    "exit_date_target": exit_target,
                    "half_size": bear_market,
                })

        # Track portfolio value
        total_value = capital
        for pos in active_positions:
            t = pos["ticker"]
            if t in all_signals and date in all_signals[t].index:
                p = all_signals[t].loc[date, "close"]
                if not pd.isna(p) and p > 0:
                    total_value += pos["shares"] * p
                else:
                    total_value += pos["shares"] * pos["entry_price"]
            else:
                total_value += pos["shares"] * pos["entry_price"]

        portfolio_value.append(total_value)
        portfolio_dates.append(date)

    # Close remaining positions at last available prices
    for pos in active_positions:
        ticker = pos["ticker"]
        if ticker in all_signals and len(all_signals[ticker]) > 0:
            exit_price = all_signals[ticker]["close"].iloc[-1]
        else:
            exit_price = pos["entry_price"]
        if pd.isna(exit_price) or exit_price <= 0:
            exit_price = pos["entry_price"]
        slippage = exit_price * SLIPPAGE_BPS / 10000
        net_exit = exit_price - slippage
        pnl = (net_exit - pos["entry_price"]) * pos["shares"]
        ret = (net_exit / pos["entry_price"]) - 1.0
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": portfolio_dates[-1].strftime("%Y-%m-%d") if portfolio_dates else "unknown",
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(net_exit, 2),
            "shares": round(pos["shares"], 4),
            "pnl": round(pnl, 2),
            "return": round(ret, 4),
            "half_size": pos["half_size"],
        })

    # Compute metrics
    pv = pd.Series(portfolio_value, index=portfolio_dates)
    daily_returns = pv.pct_change().dropna()

    # Classify each day as bull/bear based on SPY vs 200-SMA
    bull_returns = []
    bear_returns = []
    for dt, ret in daily_returns.items():
        if dt in spy_sma200.index and not pd.isna(spy_sma200.loc[dt]):
            if spy_close.loc[dt] >= spy_sma200.loc[dt]:
                bull_returns.append(ret)
            else:
                bear_returns.append(ret)
        else:
            bull_returns.append(ret)  # default to bull if no data

    trade_returns = [t["return"] for t in trades]
    n_trades = len(trades)

    result = compute_metrics(daily_returns, trade_returns, bull_returns, bear_returns, n_trades, label, pv)
    result["trades"] = trades
    result["n_trades"] = n_trades

    return result


def compute_metrics(
    daily_returns: pd.Series,
    trade_returns: list,
    bull_returns: list,
    bear_returns: list,
    n_trades: int,
    label: str,
    portfolio_series: pd.Series,
) -> dict:
    """Compute strategy performance metrics."""
    if len(daily_returns) < 10:
        return {"label": label, "error": "insufficient data", "gates": {}}

    ann_factor = np.sqrt(252)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else std_ret
    sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0

    # Max drawdown
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate and profit factor
    if trade_returns:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / len(trade_returns)
        total_gain = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")
        avg_trade_return = np.mean(trade_returns)
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_return = 0

    # Regime sharpe
    bull_sharpe = 0
    bear_sharpe = 0
    if bull_returns:
        br = pd.Series(bull_returns)
        bull_sharpe = (br.mean() / br.std() * ann_factor) if br.std() > 0 else 0
    if bear_returns:
        br = pd.Series(bear_returns)
        bear_sharpe = (br.mean() / br.std() * ann_factor) if br.std() > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    # Total return
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1.0 if len(portfolio_series) > 1 else 0

    # CAGR
    days = (portfolio_series.index[-1] - portfolio_series.index[0]).days if len(portfolio_series) > 1 else 1
    years = days / 365.25
    cagr = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    # Gates
    gates = {
        "gate1_sharpe_gt_05": {"pass": sharpe > 0.5, "value": round(sharpe, 3)},
        "gate2_permutation_p_lt_05": {"pass": None, "value": None},  # filled later
        "gate3_regime_gap_lt_05": {"pass": regime_gap < 0.5, "value": round(regime_gap, 3)},
        "gate4_maxdd_gt_neg50": {"pass": max_dd > -0.50, "value": round(max_dd, 4)},
        "gate5_min_20_trades": {"pass": n_trades >= 20, "value": n_trades},
    }

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_trade_return": round(avg_trade_return, 4),
        "n_trades": n_trades,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "gates": gates,
    }


# ─── Permutation test ───────────────────────────────────────────────────────

def permutation_test(daily_returns: pd.Series, n_iters: int = PERMUTATION_ITERS) -> float:
    """Shuffle daily returns and compute fraction of shuffled Sharpes >= observed."""
    if len(daily_returns) < 20:
        return 1.0

    ann = np.sqrt(252)
    observed_sharpe = daily_returns.mean() / daily_returns.std() * ann if daily_returns.std() > 0 else 0
    returns_arr = daily_returns.values.copy()

    count_ge = 0
    rng = np.random.default_rng(42)
    for _ in range(n_iters):
        shuffled = rng.permutation(returns_arr)
        s_std = shuffled.std()
        if s_std > 0:
            s_sharpe = shuffled.mean() / s_std * ann
        else:
            s_sharpe = 0
        if s_sharpe >= observed_sharpe:
            count_ge += 1

    return count_ge / n_iters


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Institutional Momentum / 13F Following Strategy Backtest")
    log.info("=" * 70)

    # Download data
    all_data = download_data(SP500_SAMPLE + ["SPY"], DATA_START, OOT_END)
    spy_data = all_data.pop("SPY", None)
    if spy_data is None or len(spy_data) < 200:
        log.error("Failed to download SPY data")
        return

    # Compute signals for all stocks
    log.info("Computing signals for all stocks...")
    all_signals = {}
    for ticker, df in all_data.items():
        try:
            sig = compute_signals(df)
            if len(sig) > 50:
                all_signals[ticker] = sig
        except Exception as e:
            log.warning(f"  {ticker}: signal computation failed: {e}")

    log.info(f"Signals computed for {len(all_signals)} stocks")

    # Get earnings dates for variant C
    log.info("Fetching earnings dates for variant C...")
    earnings_dates = {}
    for ticker in list(all_signals.keys())[:50]:  # limit API calls
        edates = get_earnings_dates(ticker)
        if edates:
            earnings_dates[ticker] = edates

    log.info(f"Got earnings dates for {len(earnings_dates)} stocks")

    # Run variants
    variants = {
        "A_base_2sig_40d": {"min_signals": 2, "hold_days": 40},
        "B_strict_3sig_40d": {"min_signals": 3, "hold_days": 40},
        "C_earnings_2sig_40d": {"min_signals": 2, "hold_days": 40, "earnings_filter": True},
        "D_rsi_filter_2sig_40d": {"min_signals": 2, "hold_days": 40, "require_rsi_above_50": True},
        "E_short_2sig_20d": {"min_signals": 2, "hold_days": 20},
    }

    results = {}
    for name, params in variants.items():
        result = run_backtest(
            all_signals=all_signals,
            spy_data=spy_data,
            min_signals=params.get("min_signals", 2),
            hold_days=params.get("hold_days", 40),
            require_rsi_above_50=params.get("require_rsi_above_50", False),
            earnings_filter=params.get("earnings_filter", False),
            earnings_dates=earnings_dates if params.get("earnings_filter", False) else None,
            label=name,
        )
        results[name] = result

    # Permutation tests
    log.info("Running permutation tests (100 iterations each)...")
    for name, result in results.items():
        if "error" in result:
            continue
        trades = result.get("trades", [])
        if not trades:
            result["gates"]["gate2_permutation_p_lt_05"] = {"pass": False, "value": 1.0}
            continue

        # Reconstruct daily portfolio returns from trades
        trade_rets = pd.Series([t["return"] for t in trades])
        # Use trade-level returns as proxy for permutation test
        p_value = permutation_test(trade_rets, PERMUTATION_ITERS)
        result["gates"]["gate2_permutation_p_lt_05"] = {"pass": p_value < 0.05, "value": round(p_value, 3)}
        result["permutation_p"] = round(p_value, 3)
        log.info(f"  {name}: permutation p = {p_value:.3f}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("5-GATE VALIDATION RESULTS")
    log.info("=" * 70)

    summary = {}
    for name, result in results.items():
        if "error" in result:
            log.info(f"\n{name}: ERROR - {result['error']}")
            summary[name] = result
            continue

        gates = result["gates"]
        gates_passed = sum(1 for g in gates.values() if g.get("pass") is True)
        total_gates = len(gates)

        log.info(f"\n{'─' * 50}")
        log.info(f"Variant: {name}")
        log.info(f"{'─' * 50}")
        log.info(f"  Sharpe:        {result['sharpe']}")
        log.info(f"  Sortino:       {result['sortino']}")
        log.info(f"  CAGR:          {result['cagr']:.2%}")
        log.info(f"  Total Return:  {result['total_return']:.2%}")
        log.info(f"  Max Drawdown:  {result['max_drawdown']:.2%}")
        log.info(f"  Win Rate:      {result['win_rate']:.1%}")
        log.info(f"  Profit Factor: {result['profit_factor']}")
        log.info(f"  Avg Trade Ret: {result['avg_trade_return']:.2%}")
        log.info(f"  Trades:        {result['n_trades']}")
        log.info(f"  Bull Sharpe:   {result['bull_sharpe']}")
        log.info(f"  Bear Sharpe:   {result['bear_sharpe']}")
        log.info(f"  Regime Gap:    {result['regime_gap']}")
        log.info(f"  Perm p-value:  {result.get('permutation_p', 'N/A')}")
        log.info(f"  Gates Passed:  {gates_passed}/{total_gates}")
        for gname, gval in gates.items():
            status = "PASS" if gval.get("pass") else "FAIL"
            log.info(f"    {gname}: {status} (value={gval.get('value')})")

        # Strip trades from summary (too verbose for JSON)
        summary[name] = {k: v for k, v in result.items() if k != "trades"}
        summary[name]["gates_passed"] = gates_passed
        summary[name]["gates_total"] = total_gates
        summary[name]["all_gates_pass"] = gates_passed == total_gates
        summary[name]["sample_trades"] = result.get("trades", [])[:5]

    # Save results
    output = {
        "strategy": "Institutional Momentum / 13F Following",
        "description": "Proxy-based institutional accumulation detection using volume-price signals",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "universe": f"S&P 500 sample ({len(all_signals)} stocks)",
        "cost_model": f"$0 commission, {SLIPPAGE_BPS} bps slippage",
        "starting_capital": STARTING_CAPITAL,
        "run_timestamp": datetime.now().isoformat(),
        "variants": summary,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_PATH}")

    # Final verdict
    log.info("\n" + "=" * 70)
    log.info("FINAL VERDICT")
    log.info("=" * 70)
    for name, s in summary.items():
        if "error" in s:
            log.info(f"  {name}: ERROR")
        else:
            verdict = "PASS ALL GATES" if s.get("all_gates_pass") else f"FAIL ({s.get('gates_passed')}/{s.get('gates_total')} gates)"
            log.info(f"  {name}: {verdict} | Sharpe={s.get('sharpe')} | WR={s.get('win_rate')} | DD={s.get('max_drawdown')}")


if __name__ == "__main__":
    main()
