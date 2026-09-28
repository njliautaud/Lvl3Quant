#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest
=================================
Combines 4 validated strategies for a $669 Robinhood cash account:
  1. Vol-Adj Rotation (monthly)
  2. RSI B (mean reversion)
  3. Adaptive RSI E (vol-regime bucketed)
  4. Earnings Surprise Momentum (event-driven)

Produces: signal overlap analysis, priority-based combined backtest,
combined equity curve metrics, permutation test, monthly allocation heatmap.
"""

import json
import warnings
import itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
GROWTH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "SHOP", "SQ", "SNOW", "PLTR", "COIN",
    "MELI", "MDB", "DDOG", "TTD",
]
SAFE_HAVENS = ["GLD", "TLT", "UUP", "SHY"]
ALL_TICKERS = list(set(GROWTH_UNIVERSE + SAFE_HAVENS + ["SPY", "^VIX"]))

START_DATE = "2022-01-01"
END_DATE = "2026-07-30"
STARTING_CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02%
MAX_CONCURRENT = 1  # test with 1 position at a time

DEFAULT_PRIORITY = ["earnings_surprise", "adaptive_rsi_e", "rsi_b", "vol_adj_rotation"]

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    """Download all needed price data via yfinance."""
    print("Downloading price data...")
    tickers_to_download = [t for t in ALL_TICKERS]

    data = {}
    # Download in batches to avoid issues
    batch_size = 10
    for i in range(0, len(tickers_to_download), batch_size):
        batch = tickers_to_download[i:i+batch_size]
        batch_str = " ".join(batch)
        try:
            df = yf.download(batch_str, start=START_DATE, end=END_DATE,
                           auto_adjust=True, progress=False)
            if len(batch) == 1:
                # Single ticker returns simple DataFrame
                ticker = batch[0]
                if not df.empty:
                    data[ticker] = df
            else:
                # Multi-ticker returns MultiIndex columns
                for ticker in batch:
                    try:
                        ticker_df = df.xs(ticker, level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
                        if not ticker_df.empty and not ticker_df["Close"].isna().all():
                            data[ticker] = ticker_df
                    except (KeyError, Exception):
                        pass
        except Exception as e:
            print(f"  Warning: batch download failed for {batch}: {e}")

    # Ensure SPY and VIX are present
    for critical in ["SPY", "^VIX"]:
        if critical not in data:
            try:
                df = yf.download(critical, start=START_DATE, end=END_DATE,
                               auto_adjust=True, progress=False)
                if not df.empty:
                    data[critical] = df
            except Exception:
                pass

    print(f"  Loaded data for {len(data)} tickers")
    return data


def build_price_matrix(data):
    """Build aligned close price matrix."""
    closes = {}
    for ticker, df in data.items():
        if "Close" in df.columns:
            closes[ticker] = df["Close"]

    price_df = pd.DataFrame(closes)
    price_df = price_df.ffill().dropna(how="all")
    return price_df


# ==============================================================================
# INDICATORS
# ==============================================================================
def rsi(series, period=5):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma(series, period):
    return series.rolling(period).mean()


def annualized_vol(series, window=20):
    """20-day annualized vol."""
    return series.pct_change().rolling(window).std() * np.sqrt(252) * 100


def relative_strength_3m(series):
    """3-month (63 trading days) return."""
    return series.pct_change(63)


# ==============================================================================
# STRATEGY 1: VOL-ADJ ROTATION
# ==============================================================================
def strategy_vol_adj_rotation(data, price_df):
    """Monthly rotation among growth + safe havens. Rank by RS/vol. Kill switch: VIX>20 AND SPY<50SMA."""
    rotation_universe = [t for t in GROWTH_UNIVERSE + SAFE_HAVENS if t in price_df.columns]

    spy = price_df["SPY"] if "SPY" in price_df.columns else None
    vix = price_df["^VIX"] if "^VIX" in price_df.columns else None

    trades = []

    # Get month-end dates for rebalancing
    monthly_dates = price_df.resample("ME").last().index

    for i, rebal_date in enumerate(monthly_dates):
        if rebal_date not in price_df.index:
            # Find closest date
            idx = price_df.index.searchsorted(rebal_date)
            if idx >= len(price_df.index):
                continue
            rebal_date = price_df.index[idx]

        # Kill switch: VIX > 20 AND SPY below 50-SMA
        if spy is not None and vix is not None:
            try:
                spy_val = spy.loc[:rebal_date].iloc[-1]
                vix_val = vix.loc[:rebal_date].iloc[-1]
                spy_sma50 = sma(spy, 50).loc[:rebal_date].iloc[-1]

                if not np.isnan(vix_val) and not np.isnan(spy_sma50):
                    if vix_val > 20 and spy_val < spy_sma50:
                        continue  # go to cash
            except (IndexError, KeyError):
                pass

        # Rank by 3m RS adjusted for 20d vol
        scores = {}
        for ticker in rotation_universe:
            try:
                price_series = price_df[ticker].loc[:rebal_date].dropna()
                if len(price_series) < 63:
                    continue
                rs = relative_strength_3m(price_series).iloc[-1]
                vol = annualized_vol(price_series, 20).iloc[-1]
                if np.isnan(rs) or np.isnan(vol) or vol == 0:
                    continue
                scores[ticker] = rs / vol
            except (KeyError, IndexError):
                continue

        if not scores:
            continue

        # Buy top 1
        best_ticker = max(scores, key=scores.get)

        # Hold for 1 month
        if i + 1 < len(monthly_dates):
            exit_date = monthly_dates[i + 1]
        else:
            exit_date = price_df.index[-1]

        # Find actual dates in index
        entry_idx = price_df.index.searchsorted(rebal_date)
        exit_idx = price_df.index.searchsorted(exit_date)

        if entry_idx >= len(price_df.index) or exit_idx >= len(price_df.index):
            continue

        actual_entry = price_df.index[min(entry_idx, len(price_df.index)-1)]
        actual_exit = price_df.index[min(exit_idx, len(price_df.index)-1)]

        if actual_entry >= actual_exit:
            continue

        try:
            entry_price = price_df[best_ticker].loc[actual_entry]
            exit_price = price_df[best_ticker].loc[actual_exit]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
                continue

            ret = (exit_price / entry_price - 1) - SLIPPAGE_PCT * 2

            trades.append({
                "strategy": "vol_adj_rotation",
                "ticker": best_ticker,
                "entry_date": actual_entry,
                "exit_date": actual_exit,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return": float(ret),
                "hold_days": (actual_exit - actual_entry).days,
            })
        except (KeyError, IndexError):
            continue

    return trades


# ==============================================================================
# STRATEGY 2: RSI B
# ==============================================================================
def strategy_rsi_b(data, price_df):
    """RSI(5)<20 AND price>200SMA entry. Exit RSI(5)>50 or 10 days max."""
    trades = []

    for ticker in GROWTH_UNIVERSE:
        if ticker not in price_df.columns:
            continue

        prices = price_df[ticker].dropna()
        if len(prices) < 200:
            continue

        rsi_vals = rsi(prices, 5)
        sma_200 = sma(prices, 200)

        i = 200  # start after SMA-200 is valid
        while i < len(prices):
            date = prices.index[i]

            # Entry condition
            if rsi_vals.iloc[i] < 20 and prices.iloc[i] > sma_200.iloc[i]:
                entry_price = prices.iloc[i]
                entry_date = date

                # Find exit
                exit_idx = None
                for j in range(i + 1, min(i + 11, len(prices))):  # max 10 days
                    if rsi_vals.iloc[j] > 50:
                        exit_idx = j
                        break

                if exit_idx is None:
                    exit_idx = min(i + 10, len(prices) - 1)

                exit_price = prices.iloc[exit_idx]
                exit_date = prices.index[exit_idx]

                ret = (exit_price / entry_price - 1) - SLIPPAGE_PCT * 2

                trades.append({
                    "strategy": "rsi_b",
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(ret),
                    "hold_days": (exit_date - entry_date).days,
                })

                i = exit_idx + 1  # skip to after exit
            else:
                i += 1

    return trades


# ==============================================================================
# STRATEGY 3: ADAPTIVE RSI E (Vol-Regime Bucketed)
# ==============================================================================
def strategy_adaptive_rsi_e(data, price_df):
    """Like RSI B but with vol-adaptive thresholds."""
    trades = []

    for ticker in GROWTH_UNIVERSE:
        if ticker not in price_df.columns:
            continue

        prices = price_df[ticker].dropna()
        if len(prices) < 200:
            continue

        rsi_vals = rsi(prices, 5)
        sma_200 = sma(prices, 200)
        vol_20d = annualized_vol(prices, 20)

        i = 200
        while i < len(prices):
            date = prices.index[i]

            # Price must be above 200-SMA
            if prices.iloc[i] <= sma_200.iloc[i]:
                i += 1
                continue

            current_vol = vol_20d.iloc[i]
            if np.isnan(current_vol):
                i += 1
                continue

            # Vol-regime thresholds
            if current_vol < 20:
                rsi_threshold = 15
                max_hold = 15
            elif current_vol < 35:
                rsi_threshold = 20
                max_hold = 10
            else:
                rsi_threshold = 30
                max_hold = 5

            # Entry condition
            if rsi_vals.iloc[i] < rsi_threshold:
                entry_price = prices.iloc[i]
                entry_date = date

                # Find exit
                exit_idx = None
                for j in range(i + 1, min(i + max_hold + 1, len(prices))):
                    if rsi_vals.iloc[j] > 50:
                        exit_idx = j
                        break

                if exit_idx is None:
                    exit_idx = min(i + max_hold, len(prices) - 1)

                exit_price = prices.iloc[exit_idx]
                exit_date = prices.index[exit_idx]

                ret = (exit_price / entry_price - 1) - SLIPPAGE_PCT * 2

                trades.append({
                    "strategy": "adaptive_rsi_e",
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(ret),
                    "hold_days": (exit_date - entry_date).days,
                })

                i = exit_idx + 1
            else:
                i += 1

    return trades


# ==============================================================================
# STRATEGY 4: EARNINGS SURPRISE MOMENTUM
# ==============================================================================
def get_earnings_data(ticker):
    """Get earnings dates and surprise data. Uses yfinance + synthetic fallback."""
    try:
        stock = yf.Ticker(ticker)

        # Try to get earnings dates
        try:
            earnings_df = stock.earnings_dates
            if earnings_df is not None and len(earnings_df) > 0:
                return earnings_df
        except Exception:
            pass

        # Try quarterly earnings
        try:
            earnings = stock.quarterly_earnings
            if earnings is not None and len(earnings) > 0:
                return earnings
        except Exception:
            pass
    except Exception:
        pass

    return None


def strategy_earnings_surprise(data, price_df):
    """After earnings, if EPS beat >10% AND gap up >3%, buy and hold 40 days."""
    trades = []

    spy = price_df["SPY"] if "SPY" in price_df.columns else None
    vix = price_df["^VIX"] if "^VIX" in price_df.columns else None

    for ticker in GROWTH_UNIVERSE:
        if ticker not in price_df.columns:
            continue

        prices = price_df[ticker].dropna()
        if len(prices) < 50:
            continue

        # Try to get real earnings data
        earnings_info = get_earnings_data(ticker)

        # Build list of earnings dates to check
        earnings_dates = []

        if earnings_info is not None and len(earnings_info) > 0:
            if hasattr(earnings_info, 'index'):
                for dt in earnings_info.index:
                    if isinstance(dt, pd.Timestamp):
                        d = dt.tz_localize(None) if dt.tzinfo else dt
                        if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE):
                            earnings_dates.append(d)

        # Synthetic fallback: quarterly patterns (mid-Jan, mid-Apr, mid-Jul, mid-Oct)
        if len(earnings_dates) < 4:
            for year in range(2022, 2027):
                for month in [1, 4, 7, 10]:
                    # Vary day by ticker hash to spread them out
                    day = 15 + (hash(ticker) % 15)
                    try:
                        d = pd.Timestamp(year, month, min(day, 28))
                        if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE):
                            earnings_dates.append(d)
                    except Exception:
                        pass

        earnings_dates = sorted(set(earnings_dates))

        for edate in earnings_dates:
            # Find the closest trading day
            idx = prices.index.searchsorted(edate)
            if idx >= len(prices.index) - 2 or idx < 1:
                continue

            # Check kill switch
            if spy is not None and vix is not None:
                try:
                    spy_val = spy.loc[:prices.index[idx]].iloc[-1]
                    vix_val = vix.loc[:prices.index[idx]].iloc[-1]
                    spy_sma50 = sma(spy, 50).loc[:prices.index[idx]].iloc[-1]
                    if not np.isnan(vix_val) and not np.isnan(spy_sma50):
                        if vix_val > 20 and spy_val < spy_sma50:
                            continue
                except (IndexError, KeyError):
                    pass

            # Check gap up > 3%
            pre_price = prices.iloc[idx - 1]
            post_price = prices.iloc[idx]

            if pre_price == 0 or np.isnan(pre_price) or np.isnan(post_price):
                continue

            gap = (post_price / pre_price - 1)

            if gap <= 0.03:
                continue

            # Simulate EPS beat > 10%: use gap size as proxy
            # Gap > 3% after earnings is a reasonable proxy for positive surprise
            # We'll use gap > 5% as a stricter filter to approximate 10% EPS beat
            if gap <= 0.05:
                continue

            # Buy and hold 40 days
            entry_idx = idx
            exit_idx = min(idx + 40, len(prices) - 1)

            entry_price = prices.iloc[entry_idx]
            exit_price = prices.iloc[exit_idx]
            entry_date = prices.index[entry_idx]
            exit_date = prices.index[exit_idx]

            ret = (exit_price / entry_price - 1) - SLIPPAGE_PCT * 2

            trades.append({
                "strategy": "earnings_surprise",
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return": float(ret),
                "hold_days": (exit_date - entry_date).days,
                "gap_pct": float(gap),
            })

    return trades


# ==============================================================================
# SIGNAL OVERLAP ANALYSIS
# ==============================================================================
def analyze_signal_overlap(all_trades):
    """How often do multiple strategies fire on the same day?"""
    # Build daily signal matrix
    all_dates = set()
    strategy_dates = {}

    for strat_name in DEFAULT_PRIORITY:
        strat_trades = [t for t in all_trades if t["strategy"] == strat_name]
        dates = set()
        for t in strat_trades:
            entry = t["entry_date"]
            exit_d = t["exit_date"]
            # Mark all days the position is held
            current = entry
            while current <= exit_d:
                dates.add(current)
                current += timedelta(days=1)
            all_dates.update(dates)
        strategy_dates[strat_name] = dates

    if not all_dates:
        return {"overlap_pct": 0, "correlation_matrix": {}}

    # Count overlapping days
    overlap_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for date in all_dates:
        n = sum(1 for s in strategy_dates.values() if date in s)
        overlap_counts[min(n, 4)] = overlap_counts.get(min(n, 4), 0) + 1

    total_days = len(all_dates)
    multi_signal_days = sum(v for k, v in overlap_counts.items() if k >= 2)

    # Correlation between trade returns
    strategy_returns = {}
    for strat_name in DEFAULT_PRIORITY:
        strat_trades = [t for t in all_trades if t["strategy"] == strat_name]
        if strat_trades:
            # Build monthly return series
            monthly_rets = {}
            for t in strat_trades:
                month_key = t["entry_date"].strftime("%Y-%m")
                if month_key not in monthly_rets:
                    monthly_rets[month_key] = 0
                monthly_rets[month_key] += t["return"]
            strategy_returns[strat_name] = monthly_rets

    # Compute pairwise correlations
    corr_matrix = {}
    strat_names = list(strategy_returns.keys())
    all_months = sorted(set().union(*[set(v.keys()) for v in strategy_returns.values()]))

    for s1 in strat_names:
        corr_matrix[s1] = {}
        for s2 in strat_names:
            r1 = [strategy_returns[s1].get(m, 0) for m in all_months]
            r2 = [strategy_returns[s2].get(m, 0) for m in all_months]
            if len(r1) > 2:
                corr, _ = stats.pearsonr(r1, r2)
                corr_matrix[s1][s2] = round(float(corr), 3)
            else:
                corr_matrix[s1][s2] = 0.0

    return {
        "total_signal_days": total_days,
        "overlap_counts": {str(k): v for k, v in overlap_counts.items()},
        "multi_signal_pct": round(multi_signal_days / max(total_days, 1) * 100, 1),
        "correlation_matrix": corr_matrix,
    }


# ==============================================================================
# COMBINED PORTFOLIO BACKTEST
# ==============================================================================
def run_combined_backtest(all_trades, priority_order, price_df, max_concurrent=1):
    """Run combined portfolio with priority-based signal selection."""

    if not all_trades:
        return [], pd.Series(dtype=float)

    # Sort all trades by entry date
    sorted_trades = sorted(all_trades, key=lambda t: t["entry_date"])

    # Priority map
    priority_map = {name: i for i, name in enumerate(priority_order)}

    capital = STARTING_CAPITAL
    equity_curve = {sorted_trades[0]["entry_date"] - timedelta(days=1): capital}
    executed_trades = []
    active_positions = []  # list of (trade, shares, entry_capital)

    # Build daily timeline
    all_dates = sorted(price_df.index)
    start_idx = 0

    for date in all_dates:
        # Close expired positions
        new_active = []
        for pos in active_positions:
            trade, shares, entry_cap = pos
            if date >= trade["exit_date"]:
                # Exit
                exit_val = entry_cap * (1 + trade["return"])
                capital += exit_val
                executed_trades.append({
                    **trade,
                    "shares": shares,
                    "pnl": exit_val - entry_cap,
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check for new entries today
        if len(active_positions) < max_concurrent:
            # Find all signals firing today
            candidates = [
                t for t in sorted_trades
                if t["entry_date"] == date
                and not any(t is pos[0] for pos in active_positions)
            ]

            if candidates:
                # Sort by priority
                candidates.sort(key=lambda t: priority_map.get(t["strategy"], 99))

                slots_available = max_concurrent - len(active_positions)
                for t in candidates[:slots_available]:
                    if capital > 1:  # need some capital
                        invest = capital  # all-in for single position
                        if max_concurrent > 1:
                            invest = capital / max_concurrent

                        shares = invest / t["entry_price"]
                        capital -= invest
                        active_positions.append((t, shares, invest))

        # Record equity
        total_equity = capital
        for pos in active_positions:
            trade, shares, entry_cap = pos
            # Mark to market
            if trade["ticker"] in price_df.columns:
                try:
                    current_price = price_df[trade["ticker"]].loc[:date].iloc[-1]
                    if not np.isnan(current_price):
                        total_equity += shares * current_price
                    else:
                        total_equity += entry_cap
                except (IndexError, KeyError):
                    total_equity += entry_cap
            else:
                total_equity += entry_cap

        equity_curve[date] = total_equity

    # Close any remaining positions
    for pos in active_positions:
        trade, shares, entry_cap = pos
        exit_val = entry_cap * (1 + trade["return"])
        capital += exit_val
        executed_trades.append({
            **trade,
            "shares": shares,
            "pnl": exit_val - entry_cap,
        })

    equity_series = pd.Series(equity_curve).sort_index()

    return executed_trades, equity_series


# ==============================================================================
# METRICS
# ==============================================================================
def compute_metrics(executed_trades, equity_series, price_df):
    """Compute comprehensive performance metrics."""
    if not executed_trades or equity_series.empty:
        return {"error": "No trades executed"}

    returns = equity_series.pct_change().dropna()
    returns = returns[returns != 0]  # remove days with no change

    if len(returns) < 2:
        return {"error": "Insufficient return data"}

    # Basic metrics
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100

    # Annualized return
    n_years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    ann_return = ((1 + total_return/100) ** (1 / max(n_years, 0.01)) - 1) * 100

    # Sharpe (annualized, rf=0)
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else daily_returns.std()
    sortino = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Max Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() * 100

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate and Profit Factor
    trade_returns = [t["return"] for t in executed_trades]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    win_rate = len(wins) / max(len(trade_returns), 1) * 100

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Trades per month
    n_months = max(n_years * 12, 1)
    trades_per_month = len(executed_trades) / n_months

    # Time in market
    total_trading_days = len(price_df.index)
    days_in_market = sum(t.get("hold_days", 0) for t in executed_trades)
    time_in_market = min(days_in_market / max(total_trading_days, 1) * 100, 100)

    # Per-regime performance
    spy = price_df["SPY"] if "SPY" in price_df.columns else None
    regime_metrics = compute_regime_metrics(executed_trades, equity_series, spy)

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "win_rate_pct": round(win_rate, 1),
        "max_drawdown_pct": round(float(max_dd), 2),
        "calmar": round(float(calmar), 3),
        "total_trades": len(executed_trades),
        "trades_per_month": round(trades_per_month, 2),
        "time_in_market_pct": round(time_in_market, 1),
        "final_equity": round(float(equity_series.iloc[-1]), 2),
        "starting_capital": STARTING_CAPITAL,
        "avg_trade_return_pct": round(np.mean(trade_returns) * 100, 3),
        "avg_hold_days": round(np.mean([t.get("hold_days", 0) for t in executed_trades]), 1),
        **regime_metrics,
    }


def compute_regime_metrics(executed_trades, equity_series, spy):
    """Bull = SPY above 200-SMA, Bear = below."""
    if spy is None or len(spy) < 200:
        return {"regime_gap": "N/A", "sharpe_bull": "N/A", "sharpe_bear": "N/A"}

    spy_sma200 = sma(spy, 200)

    bull_returns = []
    bear_returns = []

    for trade in executed_trades:
        entry_date = trade["entry_date"]
        try:
            spy_val = spy.loc[:entry_date].iloc[-1]
            sma_val = spy_sma200.loc[:entry_date].iloc[-1]

            if np.isnan(spy_val) or np.isnan(sma_val):
                continue

            if spy_val > sma_val:
                bull_returns.append(trade["return"])
            else:
                bear_returns.append(trade["return"])
        except (IndexError, KeyError):
            continue

    sharpe_bull = (np.mean(bull_returns) / np.std(bull_returns)) * np.sqrt(252/10) if len(bull_returns) > 1 and np.std(bull_returns) > 0 else 0
    sharpe_bear = (np.mean(bear_returns) / np.std(bear_returns)) * np.sqrt(252/10) if len(bear_returns) > 1 and np.std(bear_returns) > 0 else 0

    # Regime gap metric
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
        "bull_avg_return_pct": round(np.mean(bull_returns) * 100, 3) if bull_returns else 0,
        "bear_avg_return_pct": round(np.mean(bear_returns) * 100, 3) if bear_returns else 0,
        "regime_gap": round(float(regime_gap), 3),
        "regime_gap_pass": regime_gap < 0.50,
    }


# ==============================================================================
# PERMUTATION TEST
# ==============================================================================
def permutation_test(all_trades, price_df, n_iterations=1000):
    """Shuffle priority ordering, measure p-value of default priority."""
    print(f"\nRunning permutation test ({n_iterations} iterations)...")

    # Get baseline Sharpe
    _, baseline_equity = run_combined_backtest(all_trades, DEFAULT_PRIORITY, price_df)
    baseline_daily_rets = baseline_equity.pct_change().dropna()
    baseline_sharpe = (baseline_daily_rets.mean() / baseline_daily_rets.std()) * np.sqrt(252) if baseline_daily_rets.std() > 0 else 0

    # Run permutations
    sharpes = []
    all_perms = list(itertools.permutations(DEFAULT_PRIORITY))

    rng = np.random.default_rng(42)

    for i in range(n_iterations):
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_iterations}...")

        # Random priority ordering
        perm = list(rng.choice(all_perms))

        _, eq = run_combined_backtest(all_trades, perm, price_df)
        daily_rets = eq.pct_change().dropna()
        s = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
        sharpes.append(float(s))

    # P-value: fraction of permutations that beat baseline
    p_value = sum(1 for s in sharpes if s >= baseline_sharpe) / len(sharpes)

    return {
        "baseline_sharpe": round(float(baseline_sharpe), 3),
        "mean_perm_sharpe": round(float(np.mean(sharpes)), 3),
        "std_perm_sharpe": round(float(np.std(sharpes)), 3),
        "p_value": round(float(p_value), 4),
        "best_perm_sharpe": round(float(max(sharpes)), 3),
        "worst_perm_sharpe": round(float(min(sharpes)), 3),
        "n_iterations": n_iterations,
    }


# ==============================================================================
# MONTHLY ALLOCATION
# ==============================================================================
def monthly_allocation_summary(executed_trades):
    """For each month, which strategy was active and its return."""
    monthly = {}

    for trade in executed_trades:
        month = trade["entry_date"].strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {}

        strat = trade["strategy"]
        if strat not in monthly[month]:
            monthly[month][strat] = {"trades": 0, "total_return": 0}

        monthly[month][strat]["trades"] += 1
        monthly[month][strat]["total_return"] += trade["return"]

    # Format for output
    formatted = {}
    for month in sorted(monthly.keys()):
        formatted[month] = {}
        for strat, info in monthly[month].items():
            formatted[month][strat] = {
                "trades": info["trades"],
                "return_pct": round(info["total_return"] * 100, 2),
            }

    return formatted


# ==============================================================================
# PER-STRATEGY METRICS
# ==============================================================================
def per_strategy_metrics(all_trades):
    """Compute metrics for each strategy individually."""
    results = {}
    for strat_name in DEFAULT_PRIORITY:
        strat_trades = [t for t in all_trades if t["strategy"] == strat_name]
        if not strat_trades:
            results[strat_name] = {"n_trades": 0}
            continue

        rets = [t["return"] for t in strat_trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.0001

        results[strat_name] = {
            "n_trades": len(strat_trades),
            "avg_return_pct": round(np.mean(rets) * 100, 3),
            "total_return_pct": round(sum(rets) * 100, 2),
            "win_rate_pct": round(len(wins) / max(len(rets), 1) * 100, 1),
            "profit_factor": round(gross_profit / gross_loss, 3),
            "avg_hold_days": round(np.mean([t.get("hold_days", 0) for t in strat_trades]), 1),
            "max_return_pct": round(max(rets) * 100, 2),
            "min_return_pct": round(min(rets) * 100, 2),
        }

    return results


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("MULTI-STRATEGY PORTFOLIO BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print(f"Universe: {len(GROWTH_UNIVERSE)} growth stocks + {len(SAFE_HAVENS)} safe havens")
    print("=" * 70)

    # Load data
    data = load_data()
    price_df = build_price_matrix(data)
    print(f"Price matrix: {price_df.shape[0]} days x {price_df.shape[1]} tickers")
    print(f"Date range: {price_df.index[0].strftime('%Y-%m-%d')} to {price_df.index[-1].strftime('%Y-%m-%d')}")

    # Run individual strategies
    print("\n--- Running Individual Strategies ---")

    trades_rotation = strategy_vol_adj_rotation(data, price_df)
    print(f"  Vol-Adj Rotation: {len(trades_rotation)} trades")

    trades_rsi_b = strategy_rsi_b(data, price_df)
    print(f"  RSI B: {len(trades_rsi_b)} trades")

    trades_adaptive = strategy_adaptive_rsi_e(data, price_df)
    print(f"  Adaptive RSI E: {len(trades_adaptive)} trades")

    trades_earnings = strategy_earnings_surprise(data, price_df)
    print(f"  Earnings Surprise: {len(trades_earnings)} trades")

    all_trades = trades_rotation + trades_rsi_b + trades_adaptive + trades_earnings
    print(f"\n  Total raw signals: {len(all_trades)}")

    # A) Signal overlap analysis
    print("\n--- A) Signal Overlap Analysis ---")
    overlap = analyze_signal_overlap(all_trades)
    print(f"  Total signal days: {overlap.get('total_signal_days', 0)}")
    print(f"  Multi-signal overlap: {overlap.get('multi_signal_pct', 0)}% of days")
    print("  Overlap counts:", overlap.get("overlap_counts", {}))
    if overlap.get("correlation_matrix"):
        print("  Return correlation matrix:")
        for s1, row in overlap["correlation_matrix"].items():
            vals = " | ".join(f"{s2}: {v:+.3f}" for s2, v in row.items())
            print(f"    {s1}: {vals}")

    # Per-strategy metrics
    print("\n--- Per-Strategy Standalone Metrics ---")
    strat_metrics = per_strategy_metrics(all_trades)
    for name, m in strat_metrics.items():
        if m.get("n_trades", 0) > 0:
            print(f"  {name}: {m['n_trades']} trades | WR={m['win_rate_pct']}% | "
                  f"Avg={m['avg_return_pct']:+.3f}% | PF={m['profit_factor']:.2f} | "
                  f"Avg Hold={m['avg_hold_days']:.0f}d")
        else:
            print(f"  {name}: 0 trades")

    # B) Priority-based combined backtest
    print("\n--- B) Priority-Based Combined Backtest ---")
    print(f"  Default priority: {' > '.join(DEFAULT_PRIORITY)}")

    executed_trades, equity_series = run_combined_backtest(
        all_trades, DEFAULT_PRIORITY, price_df, MAX_CONCURRENT
    )
    print(f"  Executed trades: {len(executed_trades)}")

    # C) Combined metrics
    print("\n--- C) Combined Portfolio Metrics ---")
    metrics = compute_metrics(executed_trades, equity_series, price_df)

    print(f"  Total Return:      {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Annualized Return: {metrics.get('annualized_return_pct', 0):+.2f}%")
    print(f"  Sharpe:            {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino:           {metrics.get('sortino', 0):.3f}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.3f}")
    print(f"  Win Rate:          {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Calmar:            {metrics.get('calmar', 0):.3f}")
    print(f"  Total Trades:      {metrics.get('total_trades', 0)}")
    print(f"  Trades/Month:      {metrics.get('trades_per_month', 0):.2f}")
    print(f"  Time in Market:    {metrics.get('time_in_market_pct', 0):.1f}%")
    print(f"  Final Equity:      ${metrics.get('final_equity', 0):.2f}")
    print(f"  Avg Trade Return:  {metrics.get('avg_trade_return_pct', 0):+.3f}%")
    print(f"  Avg Hold Days:     {metrics.get('avg_hold_days', 0):.1f}")

    print(f"\n  --- Regime Analysis ---")
    print(f"  Bull Sharpe:       {metrics.get('sharpe_bull', 'N/A')}")
    print(f"  Bear Sharpe:       {metrics.get('sharpe_bear', 'N/A')}")
    print(f"  Bull Trades:       {metrics.get('bull_trades', 'N/A')}")
    print(f"  Bear Trades:       {metrics.get('bear_trades', 'N/A')}")
    print(f"  Bull Avg Return:   {metrics.get('bull_avg_return_pct', 'N/A')}%")
    print(f"  Bear Avg Return:   {metrics.get('bear_avg_return_pct', 'N/A')}%")
    print(f"  Regime Gap:        {metrics.get('regime_gap', 'N/A')} {'PASS' if metrics.get('regime_gap_pass') else 'FAIL'}")

    # Test alternative priorities
    print("\n  --- Alternative Priority Orderings ---")
    alt_priorities = [
        ["adaptive_rsi_e", "earnings_surprise", "rsi_b", "vol_adj_rotation"],
        ["rsi_b", "adaptive_rsi_e", "earnings_surprise", "vol_adj_rotation"],
        ["vol_adj_rotation", "earnings_surprise", "adaptive_rsi_e", "rsi_b"],
    ]

    alt_results = {}
    for alt in alt_priorities:
        alt_exec, alt_eq = run_combined_backtest(all_trades, alt, price_df, MAX_CONCURRENT)
        alt_daily = alt_eq.pct_change().dropna()
        alt_sharpe = (alt_daily.mean() / alt_daily.std()) * np.sqrt(252) if alt_daily.std() > 0 else 0
        alt_ret = (alt_eq.iloc[-1] / alt_eq.iloc[0] - 1) * 100
        label = " > ".join(alt)
        alt_results[label] = {"sharpe": round(float(alt_sharpe), 3), "return_pct": round(float(alt_ret), 2), "trades": len(alt_exec)}
        print(f"    {label}")
        print(f"      Sharpe={alt_sharpe:.3f} | Return={alt_ret:+.2f}% | Trades={len(alt_exec)}")

    # D) Permutation test
    perm_results = permutation_test(all_trades, price_df, n_iterations=1000)
    print(f"\n--- D) Permutation Test (n={perm_results['n_iterations']}) ---")
    print(f"  Baseline Sharpe:   {perm_results['baseline_sharpe']:.3f}")
    print(f"  Mean Perm Sharpe:  {perm_results['mean_perm_sharpe']:.3f}")
    print(f"  Std Perm Sharpe:   {perm_results['std_perm_sharpe']:.3f}")
    print(f"  P-value:           {perm_results['p_value']:.4f}")
    print(f"  Best Perm Sharpe:  {perm_results['best_perm_sharpe']:.3f}")
    print(f"  Worst Perm Sharpe: {perm_results['worst_perm_sharpe']:.3f}")

    # E) Monthly allocation
    print("\n--- E) Monthly Allocation Summary ---")
    monthly = monthly_allocation_summary(executed_trades)

    print(f"  {'Month':<8} | {'Strategy':<20} | {'Trades':<6} | {'Return':<8}")
    print(f"  {'-'*50}")
    for month in sorted(monthly.keys()):
        for strat, info in monthly[month].items():
            print(f"  {month:<8} | {strat:<20} | {info['trades']:<6} | {info['return_pct']:+.2f}%")

    # Compile full results
    results = {
        "metadata": {
            "backtest_period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "max_concurrent_positions": MAX_CONCURRENT,
            "growth_universe": GROWTH_UNIVERSE,
            "safe_havens": SAFE_HAVENS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "per_strategy_metrics": strat_metrics,
        "signal_overlap_analysis": overlap,
        "combined_metrics": metrics,
        "default_priority": DEFAULT_PRIORITY,
        "alternative_priorities": alt_results,
        "permutation_test": perm_results,
        "monthly_allocation": monthly,
    }

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/multi_strategy_portfolio_results.json")

    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            result = convert(obj)
            if result is not obj:
                return result
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  ${STARTING_CAPITAL} -> ${metrics.get('final_equity', 0):.2f} "
          f"({metrics.get('total_return_pct', 0):+.2f}%)")
    print(f"  Sharpe: {metrics.get('sharpe', 0):.3f} | Sortino: {metrics.get('sortino', 0):.3f}")
    print(f"  PF: {metrics.get('profit_factor', 0):.2f} | WR: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  MaxDD: {metrics.get('max_drawdown_pct', 0):.2f}% | Calmar: {metrics.get('calmar', 0):.3f}")
    print(f"  Trades: {metrics.get('total_trades', 0)} ({metrics.get('trades_per_month', 0):.1f}/mo)")
    print(f"  Regime Gap: {metrics.get('regime_gap', 'N/A')} ({'PASS' if metrics.get('regime_gap_pass') else 'FAIL'})")
    print(f"  Permutation p-value: {perm_results['p_value']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
