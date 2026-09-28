#!/usr/bin/env python3
"""
Pre-Earnings Announcement Drift Backtest
=========================================
Tests the well-documented tendency for stocks to drift upward in the days
before earnings announcements (option hedging flows + anticipation).

Strategy: Buy quality stocks 5-10 days before scheduled earnings, sell 1 day
before (or on earnings day for variant F) to avoid binary earnings risk.

6 Variants (A-F) with 5-gate validation.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AVGO", "CRM",
    "NFLX", "AMD", "TSLA", "ADBE", "COST", "LLY", "UNH", "JPM",
    "V", "MA", "HD", "LOW",
]

OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POS_SIZE = 200.0
MAX_CONCURRENT = 3
N_PERMUTATIONS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/pre_earnings_drift_results.json"


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def get_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download daily close prices for all tickers + SPY + ^VIX."""
    all_tickers = list(set(tickers + ["SPY", "^VIX"]))
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, progress=False)
    # yf.download returns multi-level columns: (Price, Ticker)
    # We need Close prices
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
    return closes


def get_earnings_dates(ticker: str, start: str, end: str) -> list:
    """Get historical earnings dates for a ticker using yfinance."""
    try:
        t = yf.Ticker(ticker)
        # Try to get earnings_dates (most reliable for recent history)
        try:
            ed = t.earnings_dates
            if ed is not None and len(ed) > 0:
                dates = ed.index.tz_localize(None) if ed.index.tz else ed.index
                dates = pd.DatetimeIndex(dates)
                mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                result = sorted(dates[mask].normalize().unique().tolist())
                if len(result) >= 4:
                    return result
        except Exception:
            pass

        # Try calendar
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    ed_val = cal.loc["Earnings Date"]
                    if hasattr(ed_val, "__iter__"):
                        dates = [pd.Timestamp(d).normalize() for d in ed_val]
                    else:
                        dates = [pd.Timestamp(ed_val).normalize()]
                    mask_dates = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
                    if mask_dates:
                        return sorted(mask_dates)
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed_val = cal["Earnings Date"]
                    if hasattr(ed_val, "__iter__"):
                        dates = [pd.Timestamp(d).normalize() for d in ed_val]
                    else:
                        dates = [pd.Timestamp(ed_val).normalize()]
                    mask_dates = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
                    if mask_dates:
                        return sorted(mask_dates)
        except Exception:
            pass

        # Try quarterly earnings
        try:
            qe = t.quarterly_earnings
            if qe is not None and len(qe) > 0:
                dates = pd.DatetimeIndex(qe.index)
                if dates.tz is not None:
                    dates = dates.tz_localize(None)
                dates = dates.normalize()
                mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                result = sorted(dates[mask].unique().tolist())
                if len(result) >= 4:
                    return result
        except Exception:
            pass

    except Exception:
        pass

    # Fallback: generate quarterly estimates based on typical patterns
    return _generate_quarterly_estimates(ticker, start, end)


def _generate_quarterly_estimates(ticker: str, start: str, end: str) -> list:
    """Generate estimated quarterly earnings dates when API data unavailable."""
    # Typical earnings months and approximate day offsets for major companies
    # Most report in Jan/Apr/Jul/Oct, some in Feb/May/Aug/Nov
    offsets = {
        "AAPL": (1, 27), "MSFT": (1, 24), "GOOGL": (1, 30), "META": (1, 31),
        "AMZN": (2, 1), "NVDA": (2, 21), "AVGO": (3, 7), "CRM": (3, 1),
        "NFLX": (1, 19), "AMD": (1, 31), "TSLA": (1, 25), "ADBE": (3, 15),
        "COST": (3, 7), "LLY": (2, 1), "UNH": (1, 13), "JPM": (1, 13),
        "V": (1, 26), "MA": (1, 26), "HD": (2, 21), "LOW": (2, 22),
    }
    month_offset, day = offsets.get(ticker, (1, 25))
    # Quarters: Q1 report ~month_offset, Q2 ~month_offset+3, etc.
    dates = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for q in range(4):
            m = month_offset + q * 3
            if m > 12:
                m -= 12
                y = year + 1
            else:
                y = year
            try:
                d = min(day, 28)
                ed = pd.Timestamp(y, m, d)
                if pd.Timestamp(start) <= ed <= pd.Timestamp(end):
                    dates.append(ed)
            except ValueError:
                continue
    return sorted(dates)


def get_earnings_surprise(ticker: str) -> dict:
    """Get earnings surprise history: {date: surprise_pct}."""
    try:
        t = yf.Ticker(ticker)
        try:
            eh = t.earnings_history
            if eh is not None and len(eh) > 0:
                result = {}
                for idx, row in eh.iterrows():
                    date = pd.Timestamp(idx).normalize() if not isinstance(idx, str) else pd.Timestamp(idx).normalize()
                    if "epsActual" in eh.columns and "epsEstimate" in eh.columns:
                        actual = row.get("epsActual", None)
                        est = row.get("epsEstimate", None)
                        if actual is not None and est is not None and est != 0:
                            result[date] = (actual - est) / abs(est)
                        elif actual is not None and est is not None:
                            result[date] = 1.0 if actual > 0 else -1.0
                return result
        except Exception:
            pass

        # Try quarterly_earnings for revenue surprise as proxy
        try:
            qe = t.quarterly_earnings
            if qe is not None and len(qe) > 0 and "Revenue" in qe.columns and "Earnings" in qe.columns:
                result = {}
                earnings_vals = qe["Earnings"].values
                for i, (idx, row) in enumerate(qe.iterrows()):
                    date = pd.Timestamp(idx).normalize()
                    # If earnings grew QoQ, treat as positive surprise
                    if i > 0 and earnings_vals[i-1] != 0:
                        surprise = (earnings_vals[i] - earnings_vals[i-1]) / abs(earnings_vals[i-1])
                        result[date] = surprise
                return result
        except Exception:
            pass
    except Exception:
        pass
    return {}


def compute_rsi(prices: pd.Series, window: int = 20) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ─── TRADE GENERATION ─────────────────────────────────────────────────────────

def generate_trades(
    ticker: str,
    earnings_dates: list,
    prices: pd.Series,
    trading_days: pd.DatetimeIndex,
    buy_days_before: int,
    sell_days_before: int,  # 1 = sell 1 day before earnings, 0 = sell on earnings day
    rsi_filter: bool = False,
    rsi_series: pd.Series = None,
    rsi_threshold: float = 40.0,
    vix_filter: bool = False,
    vix_series: pd.Series = None,
    vix_threshold: float = 25.0,
    surprise_filter: bool = False,
    surprise_data: dict = None,
) -> list:
    """Generate list of trades for a single ticker."""
    trades = []

    for ed in earnings_dates:
        ed = pd.Timestamp(ed)
        if ed not in trading_days:
            # Find next trading day
            future = trading_days[trading_days >= ed]
            if len(future) == 0:
                continue
            ed = future[0]

        ed_idx = trading_days.get_loc(ed)

        # Buy date: buy_days_before trading days before earnings
        buy_idx = ed_idx - buy_days_before
        if buy_idx < 0:
            continue

        # Sell date: sell_days_before trading days before earnings
        sell_idx = ed_idx - sell_days_before
        if sell_idx < 0 or sell_idx >= len(trading_days):
            continue

        buy_date = trading_days[buy_idx]
        sell_date = trading_days[sell_idx]

        if buy_date not in prices.index or sell_date not in prices.index:
            continue

        buy_price = prices.loc[buy_date]
        sell_price = prices.loc[sell_date]

        if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
            continue

        # Apply filters
        if rsi_filter and rsi_series is not None:
            if buy_date not in rsi_series.index:
                continue
            rsi_val = rsi_series.loc[buy_date]
            if pd.isna(rsi_val) or rsi_val <= rsi_threshold:
                continue

        if vix_filter and vix_series is not None:
            if buy_date not in vix_series.index:
                continue
            vix_val = vix_series.loc[buy_date]
            if pd.isna(vix_val) or vix_val >= vix_threshold:
                continue

        if surprise_filter and surprise_data is not None:
            # Find most recent earnings BEFORE this one
            prev_dates = [d for d in surprise_data.keys() if d < ed]
            if not prev_dates:
                continue
            last_surprise_date = max(prev_dates)
            if surprise_data[last_surprise_date] <= 0:
                continue  # Skip if last quarter was negative surprise

        # Apply slippage
        buy_cost = buy_price * (1 + SLIPPAGE_PCT)
        sell_proceeds = sell_price * (1 - SLIPPAGE_PCT)

        ret = (sell_proceeds - buy_cost) / buy_cost

        trades.append({
            "ticker": ticker,
            "earnings_date": ed.strftime("%Y-%m-%d"),
            "buy_date": buy_date.strftime("%Y-%m-%d"),
            "sell_date": sell_date.strftime("%Y-%m-%d"),
            "buy_price": round(float(buy_cost), 4),
            "sell_price": round(float(sell_proceeds), 4),
            "return_pct": round(float(ret) * 100, 4),
            "hold_days": int(sell_idx - buy_idx),
        })

    return trades


# ─── PORTFOLIO SIMULATION ─────────────────────────────────────────────────────

def simulate_portfolio(all_trades: list, starting_capital: float) -> dict:
    """Simulate portfolio with position sizing and concurrency limits."""
    if not all_trades:
        return {
            "equity_curve": [], "total_return": 0, "sharpe": 0,
            "max_dd": 0, "n_trades": 0, "win_rate": 0,
            "daily_returns": [],
        }

    # Sort trades by buy_date
    trades = sorted(all_trades, key=lambda x: x["buy_date"])

    # Build daily equity curve
    all_dates = set()
    for t in trades:
        all_dates.add(t["buy_date"])
        all_dates.add(t["sell_date"])
    all_dates = sorted(all_dates)

    if not all_dates:
        return {
            "equity_curve": [], "total_return": 0, "sharpe": 0,
            "max_dd": 0, "n_trades": 0, "win_rate": 0,
            "daily_returns": [],
        }

    # Track active positions per day
    capital = starting_capital
    active_positions = []  # list of (sell_date, position_size, return_pct)
    executed_trades = []
    daily_equity = []

    # Create a timeline of events
    events = []
    for t in trades:
        events.append(("buy", t["buy_date"], t))
        events.append(("sell", t["sell_date"], t))

    # Get all unique dates, sorted
    date_range = pd.bdate_range(min(all_dates), max(all_dates))
    date_strs = [d.strftime("%Y-%m-%d") for d in date_range]

    # Process day by day
    equity = starting_capital
    positions = {}  # trade_id -> {size, buy_price, sell_price, return}
    trade_id = 0
    pending_trades = list(trades)  # trades not yet entered
    pending_idx = 0

    trade_returns = []

    for date_str in date_strs:
        # Close positions that expire today
        closed_ids = []
        for tid, pos in positions.items():
            if pos["sell_date"] == date_str:
                pnl = pos["size"] * pos["return_pct"]
                equity += pos["size"] + pnl
                trade_returns.append(pos["return_pct"])
                closed_ids.append(tid)
        for tid in closed_ids:
            del positions[tid]

        # Open new positions
        while pending_idx < len(pending_trades):
            t = pending_trades[pending_idx]
            if t["buy_date"] > date_str:
                break
            if t["buy_date"] == date_str:
                if len(positions) < MAX_CONCURRENT:
                    pos_size = min(MAX_POS_SIZE, equity * 0.95)  # Keep 5% cash buffer
                    if pos_size > 10:  # Minimum viable position
                        positions[trade_id] = {
                            "size": pos_size,
                            "return_pct": t["return_pct"] / 100,
                            "sell_date": t["sell_date"],
                            "ticker": t["ticker"],
                        }
                        equity -= pos_size
                        executed_trades.append(t)
                        trade_id += 1
            pending_idx += 1

        # Record equity (mark-to-market approximate)
        total_equity = equity + sum(p["size"] for p in positions.values())
        daily_equity.append({"date": date_str, "equity": round(total_equity, 2)})

    # Calculate metrics
    if len(daily_equity) < 2:
        daily_rets = []
    else:
        eq_vals = [d["equity"] for d in daily_equity]
        daily_rets = [(eq_vals[i] - eq_vals[i-1]) / eq_vals[i-1]
                      for i in range(1, len(eq_vals)) if eq_vals[i-1] > 0]

    n_trades = len(trade_returns)
    win_rate = sum(1 for r in trade_returns if r > 0) / max(n_trades, 1)

    # Sharpe (annualized from daily returns)
    if daily_rets and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = [r for r in daily_rets if r < 0]
    if downside and np.std(downside) > 0:
        sortino = np.mean(daily_rets) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    eq_vals = [d["equity"] for d in daily_equity]
    if eq_vals:
        peak = eq_vals[0]
        max_dd = 0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    else:
        max_dd = 0

    total_return = (eq_vals[-1] / starting_capital - 1) * 100 if eq_vals else 0

    # Profit factor
    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Avg trade return
    avg_return = np.mean(trade_returns) * 100 if trade_returns else 0

    return {
        "equity_curve": daily_equity,
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "avg_return_pct": round(avg_return, 4),
        "final_equity": round(eq_vals[-1], 2) if eq_vals else starting_capital,
        "daily_returns": daily_rets,
        "trade_returns": trade_returns,
    }


# ─── VALIDATION GATES ─────────────────────────────────────────────────────────

def permutation_test(
    all_trades: list,
    actual_sharpe: float,
    prices: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    n_perms: int = 1000,
) -> float:
    """Shuffle entry dates (buy at random days instead of pre-earnings) to test significance."""
    if len(all_trades) < 5:
        return 1.0

    rng = np.random.RandomState(42)
    count_better = 0

    # Get list of all valid trading days for random entry
    valid_days = trading_days.tolist()
    n_valid = len(valid_days)

    for _ in range(n_perms):
        shuffled_trades = []
        for t in all_trades:
            hold_days = t["hold_days"]
            # Random entry date
            max_idx = n_valid - hold_days - 1
            if max_idx < 1:
                continue
            rand_idx = rng.randint(0, max_idx)
            rand_buy_date = valid_days[rand_idx]
            rand_sell_date = valid_days[rand_idx + hold_days]

            ticker = t["ticker"]
            if ticker not in prices.columns:
                continue

            ticker_prices = prices[ticker]
            if rand_buy_date not in ticker_prices.index or rand_sell_date not in ticker_prices.index:
                continue

            buy_p = ticker_prices.loc[rand_buy_date]
            sell_p = ticker_prices.loc[rand_sell_date]
            if pd.isna(buy_p) or pd.isna(sell_p) or buy_p <= 0:
                continue

            buy_cost = buy_p * (1 + SLIPPAGE_PCT)
            sell_proc = sell_p * (1 - SLIPPAGE_PCT)
            ret = (sell_proc - buy_cost) / buy_cost

            shuffled_trades.append({
                "buy_date": rand_buy_date.strftime("%Y-%m-%d") if hasattr(rand_buy_date, "strftime") else str(rand_buy_date)[:10],
                "sell_date": rand_sell_date.strftime("%Y-%m-%d") if hasattr(rand_sell_date, "strftime") else str(rand_sell_date)[:10],
                "return_pct": ret * 100,
                "hold_days": hold_days,
                "ticker": ticker,
            })

        result = simulate_portfolio(shuffled_trades, STARTING_CAPITAL)
        if result["sharpe"] >= actual_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return round(p_value, 4)


def regime_analysis(
    all_trades: list,
    spy_prices: pd.Series,
) -> dict:
    """Split trades into bull/bear regime and compute per-regime Sharpe."""
    spy_sma200 = spy_prices.rolling(200, min_periods=200).mean()

    bull_trades = []
    bear_trades = []

    for t in all_trades:
        buy_date = pd.Timestamp(t["buy_date"])
        if buy_date in spy_sma200.index:
            sma_val = spy_sma200.loc[buy_date]
            spy_val = spy_prices.loc[buy_date]
            if pd.notna(sma_val) and pd.notna(spy_val):
                if spy_val > sma_val:
                    bull_trades.append(t)
                else:
                    bear_trades.append(t)
            else:
                bull_trades.append(t)  # default to bull if no data
        else:
            bull_trades.append(t)

    bull_result = simulate_portfolio(bull_trades, STARTING_CAPITAL)
    bear_result = simulate_portfolio(bear_trades, STARTING_CAPITAL)

    bull_sharpe = bull_result["sharpe"]
    bear_sharpe = bear_result["sharpe"]

    # Regime gap: |bull - bear| / max(|bull|, |bear|)
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    if max_abs > 0:
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs
    else:
        regime_gap = 0

    return {
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "regime_gap": round(regime_gap, 4),
    }


def validate_variant(result: dict, regime: dict, p_value: float) -> dict:
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": result["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": result["max_dd_pct"] > -50,
        "min_20_trades": result["n_trades"] >= 20,
    }
    all_pass = all(gates.values())
    return {"gates": gates, "all_pass": all_pass}


# ─── VARIANT DEFINITIONS ──────────────────────────────────────────────────────

VARIANTS = {
    "A": {
        "description": "Buy 10d before, sell 1d before earnings",
        "buy_days_before": 10,
        "sell_days_before": 1,
        "rsi_filter": False,
        "vix_filter": False,
        "surprise_filter": False,
    },
    "B": {
        "description": "Buy 5d before, sell 1d before (shorter exposure)",
        "buy_days_before": 5,
        "sell_days_before": 1,
        "rsi_filter": False,
        "vix_filter": False,
        "surprise_filter": False,
    },
    "C": {
        "description": "Buy 7d before, sell 1d before, RSI>40 filter",
        "buy_days_before": 7,
        "sell_days_before": 1,
        "rsi_filter": True,
        "vix_filter": False,
        "surprise_filter": False,
    },
    "D": {
        "description": "Buy 10d before, sell 1d before, VIX<25 filter",
        "buy_days_before": 10,
        "sell_days_before": 1,
        "rsi_filter": False,
        "vix_filter": True,
        "surprise_filter": False,
    },
    "E": {
        "description": "Buy 7d before, sell 1d before, positive surprise last Q",
        "buy_days_before": 7,
        "sell_days_before": 1,
        "rsi_filter": False,
        "vix_filter": False,
        "surprise_filter": True,
    },
    "F": {
        "description": "Buy 10d before, sell ON earnings day (hold through)",
        "buy_days_before": 10,
        "sell_days_before": 0,
        "rsi_filter": False,
        "vix_filter": False,
        "surprise_filter": False,
    },
}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PRE-EARNINGS ANNOUNCEMENT DRIFT BACKTEST")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print(f"Universe: {len(TICKERS)} stocks")
    print("=" * 80)

    # Need extra history for SPY 200-SMA and RSI
    data_start = "2021-01-01"  # ~1yr before OOT for SMA200

    # Download prices
    prices = get_price_data(TICKERS, data_start, OOT_END)
    trading_days = prices.index

    # Get SPY and VIX
    spy_prices = prices["SPY"] if "SPY" in prices.columns else None
    vix_prices = prices["^VIX"] if "^VIX" in prices.columns else None

    # Compute RSI for all tickers
    rsi_data = {}
    for ticker in TICKERS:
        if ticker in prices.columns:
            rsi_data[ticker] = compute_rsi(prices[ticker], window=20)

    # Get earnings dates and surprise data
    print("\nFetching earnings dates...")
    earnings_data = {}
    surprise_data = {}
    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        earnings_data[ticker] = get_earnings_dates(ticker, OOT_START, OOT_END)
        surprise_data[ticker] = get_earnings_surprise(ticker)
        print(f"{len(earnings_data[ticker])} dates found")

    # Filter trading days to OOT period
    oot_mask = (trading_days >= pd.Timestamp(OOT_START)) & (trading_days <= pd.Timestamp(OOT_END))
    oot_trading_days = trading_days[oot_mask]

    print(f"\nOOT trading days: {len(oot_trading_days)}")
    print(f"Date range: {oot_trading_days[0].strftime('%Y-%m-%d')} to {oot_trading_days[-1].strftime('%Y-%m-%d')}")

    # Run each variant
    results = {}
    for var_name, var_config in VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"VARIANT {var_name}: {var_config['description']}")
        print(f"{'─' * 60}")

        # Generate trades for all tickers
        all_trades = []
        for ticker in TICKERS:
            if ticker not in prices.columns:
                continue

            ticker_prices = prices[ticker]
            ticker_earnings = earnings_data[ticker]

            trades = generate_trades(
                ticker=ticker,
                earnings_dates=ticker_earnings,
                prices=ticker_prices,
                trading_days=trading_days,
                buy_days_before=var_config["buy_days_before"],
                sell_days_before=var_config["sell_days_before"],
                rsi_filter=var_config["rsi_filter"],
                rsi_series=rsi_data.get(ticker),
                rsi_threshold=40.0,
                vix_filter=var_config["vix_filter"],
                vix_series=vix_prices,
                vix_threshold=25.0,
                surprise_filter=var_config["surprise_filter"],
                surprise_data=surprise_data.get(ticker, {}),
            )

            # Filter to OOT period
            trades = [t for t in trades
                      if t["buy_date"] >= OOT_START and t["sell_date"] <= OOT_END]
            all_trades.extend(trades)

        print(f"  Total trades generated: {len(all_trades)}")

        # Simulate portfolio
        sim = simulate_portfolio(all_trades, STARTING_CAPITAL)
        print(f"  Executed trades: {sim['n_trades']}")
        print(f"  Final equity: ${sim['final_equity']:.2f}")
        print(f"  Total return: {sim['total_return_pct']:.2f}%")
        print(f"  Sharpe: {sim['sharpe']:.4f}")
        print(f"  Sortino: {sim['sortino']:.4f}")
        print(f"  Win rate: {sim['win_rate']:.1f}%")
        print(f"  Max DD: {sim['max_dd_pct']:.2f}%")
        print(f"  Profit factor: {sim['profit_factor']:.4f}")

        # Regime analysis
        if spy_prices is not None:
            regime = regime_analysis(all_trades, spy_prices)
            print(f"  Bull Sharpe: {regime['bull_sharpe']:.4f} ({regime['bull_trades']} trades)")
            print(f"  Bear Sharpe: {regime['bear_sharpe']:.4f} ({regime['bear_trades']} trades)")
            print(f"  Regime gap: {regime['regime_gap']:.4f}")
        else:
            regime = {"bull_sharpe": 0, "bear_sharpe": 0, "bull_trades": 0, "bear_trades": 0, "regime_gap": 0}

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...", flush=True)
        p_value = permutation_test(all_trades, sim["sharpe"], prices, oot_trading_days, N_PERMUTATIONS)
        print(f"  Permutation p-value: {p_value:.4f}")

        # Validate
        validation = validate_variant(sim, regime, p_value)
        print(f"  Validation: {'PASS' if validation['all_pass'] else 'FAIL'}")
        for gate_name, gate_val in validation["gates"].items():
            status = "PASS" if gate_val else "FAIL"
            print(f"    {gate_name}: {status}")

        # Store results (without large arrays for JSON)
        results[var_name] = {
            "description": var_config["description"],
            "config": {k: v for k, v in var_config.items() if k != "description"},
            "n_trades": sim["n_trades"],
            "total_return_pct": sim["total_return_pct"],
            "final_equity": sim["final_equity"],
            "sharpe": sim["sharpe"],
            "sortino": sim["sortino"],
            "max_dd_pct": sim["max_dd_pct"],
            "win_rate": sim["win_rate"],
            "profit_factor": sim["profit_factor"],
            "avg_return_pct": sim["avg_return_pct"],
            "regime": regime,
            "perm_p_value": p_value,
            "validation": validation,
        }

    # ─── SUMMARY TABLE ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Var':>3} | {'Description':<55} | {'#Tr':>4} | {'Ret%':>7} | {'Sharpe':>7} | {'Sortino':>7} | {'WR%':>5} | {'MaxDD%':>7} | {'PF':>6} | {'p-val':>6} | {'RGap':>5} | {'Pass':>4}"
    print(header)
    print("-" * len(header))

    for var_name, r in results.items():
        v = r["validation"]["all_pass"]
        print(
            f"  {var_name} | {r['description']:<55} | {r['n_trades']:>4} | "
            f"{r['total_return_pct']:>6.1f}% | {r['sharpe']:>7.4f} | {r['sortino']:>7.4f} | "
            f"{r['win_rate']:>4.1f}% | {r['max_dd_pct']:>6.1f}% | {r['profit_factor']:>5.2f} | "
            f"{r['perm_p_value']:>6.4f} | {r['regime']['regime_gap']:>5.3f} | "
            f"{'YES' if v else 'NO':>4}"
        )

    # Count passing variants
    passing = [v for v, r in results.items() if r["validation"]["all_pass"]]
    print(f"\nPassing variants: {len(passing)}/{len(results)} — {', '.join(passing) if passing else 'None'}")

    # Save results
    output = {
        "strategy": "Pre-Earnings Announcement Drift",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "universe": TICKERS,
        "slippage_pct": SLIPPAGE_PCT,
        "max_position_size": MAX_POS_SIZE,
        "max_concurrent_positions": MAX_CONCURRENT,
        "n_permutations": N_PERMUTATIONS,
        "run_timestamp": dt.datetime.now().isoformat(),
        "variants": results,
    }

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
