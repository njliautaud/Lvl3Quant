#!/usr/bin/env python3
"""
Walk-Forward Momentum Backtest
===============================
Backtests 12-1 month momentum strategy with:
  - SLIDING walk-forward (252-day lookback, 21-day hold, slide 21 days) — HC #0
  - Dynamic exits: 8% stop-loss, 2x ATR(20) trailing stop, momentum breakdown — HC #684
  - Multiple portfolio sizes (top 5/10/15/20)
  - Equal-weight, monthly rebalance
  - Transaction costs: 0.1% round-trip
  - Risk-adjusted metrics: Sharpe, Sortino, MaxDD, WR, CAGR, PF
  - Regime analysis (green/red months vs SPY)
  - Benchmark: SPY buy-and-hold

Usage:
  python3 growth/momentum_backtest.py
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent dir for universe import
sys.path.insert(0, str(Path(__file__).parent))
from universe import build_universe


# ── Constants ────────────────────────────────────────────────────────────────

LOOKBACK_DAYS = 252       # 12 months of trading days
SKIP_RECENT = 21          # skip most recent month (12-1 month momentum)
HOLD_PERIOD = 21          # monthly rebalance (21 trading days)
ATR_PERIOD = 20           # ATR lookback for trailing stop
STOP_LOSS_PCT = 0.08      # 8% stop-loss from entry
TRAILING_STOP_ATR_MULT = 2.0  # 2x ATR(20) trailing stop
MOM_BREAKDOWN_DAYS = 10   # exit if 10-day momentum flips negative
COST_PER_TRADE = 0.001    # 0.1% round-trip transaction cost
PORTFOLIO_SIZES = [5, 10, 15, 20]

DATA_START = "2018-01-01"   # enough history for momentum calc
BACKTEST_START = "2019-01-01"


# ── Data Download ────────────────────────────────────────────────────────────

def download_data(tickers: list, start: str, end: str = None) -> tuple:
    """Download OHLCV data for all tickers. Returns (close_df, high_df, low_df, volume_df)."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"[INFO] Downloading data for {len(tickers)} tickers from {start} to {end}...")

    batch_size = 50
    all_close, all_high, all_low, all_volume = {}, {}, {}, {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            data = yf.download(batch_str, start=start, end=end,
                               progress=False, threads=True, group_by="ticker")
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    for ticker in batch:
                        try:
                            if ticker in data.columns.get_level_values(0):
                                td = data[ticker]
                                close_col = "Adj Close" if "Adj Close" in td.columns else "Close"
                                c = td[close_col].dropna()
                                if len(c) > 200:
                                    all_close[ticker] = c
                                    all_high[ticker] = td["High"].dropna()
                                    all_low[ticker] = td["Low"].dropna()
                                    all_volume[ticker] = td["Volume"].dropna()
                        except Exception:
                            pass
                elif len(batch) == 1:
                    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
                    c = data[close_col].dropna()
                    if len(c) > 200:
                        all_close[batch[0]] = c
                        all_high[batch[0]] = data["High"].dropna()
                        all_low[batch[0]] = data["Low"].dropna()
                        all_volume[batch[0]] = data["Volume"].dropna()

            got = sum(1 for t in batch if t in all_close)
            print(f"  Batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}: "
                  f"{got}/{len(batch)} tickers")
        except Exception as e:
            print(f"  [WARN] Batch {i // batch_size + 1} failed: {e}")

        if i + batch_size < len(tickers):
            time.sleep(0.5)

    close_df = pd.DataFrame(all_close)
    high_df = pd.DataFrame(all_high)
    low_df = pd.DataFrame(all_low)
    volume_df = pd.DataFrame(all_volume)

    # Align all DataFrames to common index
    common_idx = close_df.index
    high_df = high_df.reindex(common_idx)
    low_df = low_df.reindex(common_idx)
    volume_df = volume_df.reindex(common_idx)

    print(f"[INFO] Got data for {len(close_df.columns)} tickers, "
          f"{len(close_df)} trading days ({close_df.index[0].date()} to {close_df.index[-1].date()})")

    return close_df, high_df, low_df, volume_df


def download_spy(start: str, end: str = None) -> pd.Series:
    """Download SPY close prices for benchmark and regime classification."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    data = yf.download("SPY", start=start, end=end, progress=False)
    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    spy = data[close_col].dropna()
    # Flatten MultiIndex columns if needed
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    return spy


# ── Momentum Calculation ─────────────────────────────────────────────────────

def calc_momentum_12_1(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    12-1 month momentum at a specific index position.
    Uses data from (as_of_idx - 252) to (as_of_idx - 21).
    No lookahead: only uses data up to as_of_idx.
    """
    start_idx = max(0, as_of_idx - LOOKBACK_DAYS)
    end_idx = as_of_idx - SKIP_RECENT

    if end_idx <= start_idx:
        return pd.Series(dtype=float)

    price_start = prices.iloc[start_idx]
    price_end = prices.iloc[end_idx]

    momentum = (price_end / price_start) - 1.0
    return momentum.dropna()


def calc_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
             as_of_idx: int, period: int = 20) -> pd.Series:
    """Calculate ATR(period) as of a specific index. No lookahead."""
    start = max(0, as_of_idx - period)
    end = as_of_idx + 1  # inclusive of as_of_idx

    h = high.iloc[start:end]
    l = low.iloc[start:end]
    c = close.iloc[start:end]

    # True range
    prev_c = c.shift(1)
    tr = pd.DataFrame({
        "hl": h - l,
        "hc": (h - prev_c).abs(),
        "lc": (l - prev_c).abs(),
    }).max(axis=1) if len(c.shape) == 1 else None

    # For multi-column DataFrames
    tr_dict = {}
    for ticker in close.columns:
        try:
            hi = h[ticker].values
            lo = l[ticker].values
            cl = c[ticker].values
            prev_cl = np.roll(cl, 1)
            prev_cl[0] = cl[0]
            true_range = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
            tr_dict[ticker] = np.nanmean(true_range[1:])  # skip first (no prev close)
        except Exception:
            pass

    return pd.Series(tr_dict)


# ── Portfolio Position Tracking ──────────────────────────────────────────────

class Position:
    """Track a single position with dynamic exit logic."""
    __slots__ = ["ticker", "entry_price", "entry_date", "entry_idx",
                 "trailing_high", "shares", "cost_basis"]

    def __init__(self, ticker, entry_price, entry_date, entry_idx, shares):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_idx = entry_idx
        self.trailing_high = entry_price
        self.shares = shares
        self.cost_basis = entry_price * shares * (1 + COST_PER_TRADE)

    def check_exit(self, current_price, current_idx, atr_value, mom_10d):
        """
        Check dynamic exit conditions. Returns (should_exit, reason).
        """
        # Update trailing high
        if current_price > self.trailing_high:
            self.trailing_high = current_price

        # 1. Stop-loss: 8% from entry
        if current_price <= self.entry_price * (1 - STOP_LOSS_PCT):
            return True, "stop_loss"

        # 2. Trailing stop: 2x ATR(20) from trailing high
        if not np.isnan(atr_value) and atr_value > 0:
            trail_level = self.trailing_high - TRAILING_STOP_ATR_MULT * atr_value
            if current_price <= trail_level:
                return True, "trailing_stop"

        # 3. Momentum breakdown: 10-day momentum flips negative
        if not np.isnan(mom_10d) and mom_10d < 0:
            return True, "momentum_breakdown"

        return False, None

    def pnl(self, exit_price):
        """Calculate P&L including transaction costs."""
        gross_proceeds = exit_price * self.shares * (1 - COST_PER_TRADE)
        return gross_proceeds - self.cost_basis


# ── Walk-Forward Backtest Engine ─────────────────────────────────────────────

def run_backtest(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 spy: pd.Series, portfolio_size: int) -> dict:
    """
    Run sliding walk-forward momentum backtest for a given portfolio size.

    Walk-forward design:
      - At each rebalance date (every 21 trading days):
        - Look back 252 days for momentum calc (skip most recent 21)
        - Select top N stocks by 12-1 momentum
        - Equal-weight allocation
      - Between rebalances: check dynamic exits DAILY
      - Slide forward 21 days, repeat
    """
    dates = close.index
    n_dates = len(dates)

    # Find first valid rebalance date (need LOOKBACK_DAYS of history)
    backtest_start = pd.Timestamp(BACKTEST_START)
    first_valid_idx = 0
    for i in range(n_dates):
        if dates[i] >= backtest_start and i >= LOOKBACK_DAYS:
            first_valid_idx = i
            break

    if first_valid_idx == 0:
        print(f"[ERROR] Not enough data to start backtest at {BACKTEST_START}")
        return {}

    # Track portfolio
    cash = 1_000_000.0  # start with $1M
    initial_capital = cash
    positions = {}  # ticker -> Position
    equity_curve = []
    trade_log = []
    exit_reasons = {"stop_loss": 0, "trailing_stop": 0, "momentum_breakdown": 0, "rebalance": 0}

    # Generate rebalance dates
    rebalance_indices = list(range(first_valid_idx, n_dates, HOLD_PERIOD))

    # Walk through every trading day
    current_rebalance_ptr = 0  # pointer into rebalance_indices
    next_rebalance_idx = rebalance_indices[0] if rebalance_indices else n_dates

    for day_idx in range(first_valid_idx, n_dates):
        current_date = dates[day_idx]

        # ── Daily exit checks (before rebalance) ──
        tickers_to_exit = []
        for ticker, pos in positions.items():
            try:
                current_price = close[ticker].iloc[day_idx]
                if np.isnan(current_price):
                    tickers_to_exit.append((ticker, "missing_data"))
                    continue

                # Calc ATR for this ticker as of today
                atr_val = np.nan
                if day_idx >= ATR_PERIOD:
                    try:
                        h = high[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        l = low[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        c = close[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        prev_c = np.roll(c, 1)
                        prev_c[0] = c[0]
                        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
                        atr_val = np.nanmean(tr[1:])
                    except Exception:
                        pass

                # Calc 10-day momentum
                mom_10d = np.nan
                if day_idx >= MOM_BREAKDOWN_DAYS:
                    try:
                        p_now = close[ticker].iloc[day_idx]
                        p_10d_ago = close[ticker].iloc[day_idx - MOM_BREAKDOWN_DAYS]
                        if not np.isnan(p_10d_ago) and p_10d_ago > 0:
                            mom_10d = (p_now / p_10d_ago) - 1.0
                    except Exception:
                        pass

                should_exit, reason = pos.check_exit(current_price, day_idx, atr_val, mom_10d)
                if should_exit:
                    tickers_to_exit.append((ticker, reason))
            except Exception:
                tickers_to_exit.append((ticker, "error"))

        # Execute exits
        for ticker, reason in tickers_to_exit:
            pos = positions[ticker]
            try:
                exit_price = close[ticker].iloc[day_idx]
                if np.isnan(exit_price):
                    exit_price = pos.entry_price  # fallback
                pnl = pos.pnl(exit_price)
                cash += exit_price * pos.shares * (1 - COST_PER_TRADE)
                trade_log.append({
                    "ticker": ticker,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(current_date.date()),
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": (exit_price / pos.entry_price - 1) * 100,
                    "reason": reason,
                    "hold_days": day_idx - pos.entry_idx,
                })
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            except Exception:
                pass
            del positions[ticker]

        # ── Rebalance check ──
        is_rebalance_day = (day_idx == next_rebalance_idx)

        if is_rebalance_day:
            # Close remaining positions at rebalance (sell everything, buy new portfolio)
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                try:
                    exit_price = close[ticker].iloc[day_idx]
                    if np.isnan(exit_price):
                        exit_price = pos.entry_price
                    pnl = pos.pnl(exit_price)
                    cash += exit_price * pos.shares * (1 - COST_PER_TRADE)
                    trade_log.append({
                        "ticker": ticker,
                        "entry_date": str(pos.entry_date.date()),
                        "exit_date": str(current_date.date()),
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": (exit_price / pos.entry_price - 1) * 100,
                        "reason": "rebalance",
                        "hold_days": day_idx - pos.entry_idx,
                    })
                    exit_reasons["rebalance"] += 1
                except Exception:
                    pass
            positions.clear()

            # Calculate momentum rankings (NO LOOKAHEAD)
            momentum = calc_momentum_12_1(close, day_idx)

            # Filter: must have valid price today and enough history
            valid_tickers = []
            for ticker in momentum.index:
                try:
                    price = close[ticker].iloc[day_idx]
                    if not np.isnan(price) and price > 5.0 and not np.isnan(momentum[ticker]):
                        valid_tickers.append(ticker)
                except Exception:
                    pass

            momentum = momentum[valid_tickers].sort_values(ascending=False)

            # Select top N
            selected = momentum.head(portfolio_size).index.tolist()

            if len(selected) > 0:
                # Equal weight allocation
                per_stock_capital = cash / len(selected)

                for ticker in selected:
                    try:
                        entry_price = close[ticker].iloc[day_idx]
                        if np.isnan(entry_price) or entry_price <= 0:
                            continue
                        shares = int(per_stock_capital / (entry_price * (1 + COST_PER_TRADE)))
                        if shares > 0:
                            cost = entry_price * shares * (1 + COST_PER_TRADE)
                            cash -= cost
                            positions[ticker] = Position(
                                ticker, entry_price, current_date, day_idx, shares
                            )
                    except Exception:
                        pass

            # Advance rebalance pointer
            current_rebalance_ptr += 1
            if current_rebalance_ptr < len(rebalance_indices):
                next_rebalance_idx = rebalance_indices[current_rebalance_ptr]
            else:
                next_rebalance_idx = n_dates  # no more rebalances

        # ── Record daily equity ──
        portfolio_value = cash
        for ticker, pos in positions.items():
            try:
                price = close[ticker].iloc[day_idx]
                if not np.isnan(price):
                    portfolio_value += price * pos.shares
                else:
                    portfolio_value += pos.entry_price * pos.shares
            except Exception:
                portfolio_value += pos.entry_price * pos.shares

        equity_curve.append({
            "date": current_date,
            "equity": portfolio_value,
            "n_positions": len(positions),
        })

    # Close any remaining positions at end
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        try:
            exit_price = close[ticker].iloc[-1]
            if np.isnan(exit_price):
                exit_price = pos.entry_price
            pnl = pos.pnl(exit_price)
            trade_log.append({
                "ticker": ticker,
                "entry_date": str(pos.entry_date.date()),
                "exit_date": str(dates[-1].date()),
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "return_pct": (exit_price / pos.entry_price - 1) * 100,
                "reason": "end_of_backtest",
                "hold_days": n_dates - 1 - pos.entry_idx,
            })
        except Exception:
            pass

    return {
        "portfolio_size": portfolio_size,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "exit_reasons": exit_reasons,
        "initial_capital": initial_capital,
    }


# ── Metrics Calculation ──────────────────────────────────────────────────────

def calc_metrics(result: dict, spy: pd.Series) -> dict:
    """Calculate risk-adjusted performance metrics."""
    eq = pd.DataFrame(result["equity_curve"])
    if eq.empty:
        return {}

    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    # Daily returns
    eq["return"] = eq["equity"].pct_change()
    daily_returns = eq["return"].dropna()

    if len(daily_returns) < 30:
        return {"error": "Not enough data"}

    # Basic metrics
    total_return = eq["equity"].iloc[-1] / result["initial_capital"] - 1
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe (annualized, rf=0)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Sortino (annualized, rf=0)
    downside = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Max drawdown
    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Trade-level stats
    trades = result["trade_log"]
    if trades:
        pnls = [t["pnl"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / len(pnls) if pnls else 0
        profit_factor = sum(winners) / abs(sum(losers)) if losers and sum(losers) != 0 else float("inf")
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        avg_hold_days = np.mean([t["hold_days"] for t in trades])
    else:
        win_rate = profit_factor = avg_win = avg_loss = avg_hold_days = 0

    # ── SPY benchmark ──
    spy_aligned = spy.reindex(eq.index).dropna()
    if len(spy_aligned) > 30:
        spy_ret = spy_aligned.pct_change().dropna()
        spy_total = spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1
        spy_cagr = (1 + spy_total) ** (1 / max(n_years, 0.01)) - 1
        spy_sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252) if spy_ret.std() > 0 else 0
        spy_cummax = spy_aligned.cummax()
        spy_dd = ((spy_aligned - spy_cummax) / spy_cummax).min()
    else:
        spy_total = spy_cagr = spy_sharpe = spy_dd = 0

    # ── Regime analysis ──
    # Classify each month as green (SPY > 0) or red (SPY <= 0)
    monthly_eq = eq["equity"].resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna()

    spy_monthly = spy.resample("ME").last()
    spy_monthly_ret = spy_monthly.pct_change().dropna()

    # Align
    common_months = monthly_ret.index.intersection(spy_monthly_ret.index)
    if len(common_months) > 10:
        strat_monthly = monthly_ret.loc[common_months]
        spy_m = spy_monthly_ret.loc[common_months]

        green_mask = spy_m > 0
        red_mask = spy_m <= 0

        green_returns = strat_monthly[green_mask]
        red_returns = strat_monthly[red_mask]

        green_sharpe = (green_returns.mean() / green_returns.std() * np.sqrt(12)
                        if len(green_returns) > 3 and green_returns.std() > 0 else 0)
        red_sharpe = (red_returns.mean() / red_returns.std() * np.sqrt(12)
                      if len(red_returns) > 3 and red_returns.std() > 0 else 0)

        regime_skew = (abs(green_sharpe - red_sharpe) /
                       max(abs(green_sharpe), abs(red_sharpe), 0.01))
    else:
        green_sharpe = red_sharpe = regime_skew = 0
        green_returns = red_returns = pd.Series(dtype=float)

    return {
        "portfolio_size": result["portfolio_size"],
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "win_rate_pct": round(win_rate * 100, 1),
        "total_trades": len(trades),
        "avg_hold_days": round(avg_hold_days, 1) if trades else 0,
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "exit_reasons": result["exit_reasons"],
        "benchmark_spy": {
            "total_return_pct": round(spy_total * 100, 2) if spy_total else 0,
            "cagr_pct": round(spy_cagr * 100, 2) if spy_cagr else 0,
            "sharpe": round(spy_sharpe, 3) if spy_sharpe else 0,
            "max_drawdown_pct": round(spy_dd * 100, 2) if spy_dd else 0,
        },
        "regime_analysis": {
            "green_months": int(len(green_returns)) if isinstance(green_returns, pd.Series) else 0,
            "red_months": int(len(red_returns)) if isinstance(red_returns, pd.Series) else 0,
            "green_sharpe_monthly": round(green_sharpe, 3),
            "red_sharpe_monthly": round(red_sharpe, 3),
            "regime_skew": round(regime_skew, 3),
            "regime_agnostic": regime_skew <= 0.50,
        },
        "backtest_period": {
            "start": str(eq.index[0].date()),
            "end": str(eq.index[-1].date()),
            "trading_days": len(daily_returns),
            "years": round(n_years, 2),
        },
    }


# ── Display ──────────────────────────────────────────────────────────────────

def print_results(all_metrics: list):
    """Print formatted backtest results."""
    print(f"\n{'=' * 80}")
    print(f"  MOMENTUM BACKTEST RESULTS — Walk-Forward Sliding Window")
    print(f"  12-1 Month Momentum | Monthly Rebalance | Dynamic Exits")
    print(f"{'=' * 80}")

    if not all_metrics:
        print("[ERROR] No results to display.")
        return

    # Period info from first result
    period = all_metrics[0].get("backtest_period", {})
    print(f"\n  Period: {period.get('start', '?')} to {period.get('end', '?')} "
          f"({period.get('years', '?')} years, {period.get('trading_days', '?')} trading days)")

    # SPY benchmark
    spy_b = all_metrics[0].get("benchmark_spy", {})
    print(f"\n  SPY Buy-and-Hold: CAGR {spy_b.get('cagr_pct', 0):.1f}% | "
          f"Sharpe {spy_b.get('sharpe', 0):.3f} | "
          f"MaxDD {spy_b.get('max_drawdown_pct', 0):.1f}%")

    # Results table
    print(f"\n  {'Size':>4} | {'CAGR':>7} | {'Sharpe':>7} | {'Sortino':>8} | "
          f"{'MaxDD':>7} | {'WR':>5} | {'PF':>6} | {'Trades':>6} | {'AvgHold':>7}")
    print(f"  {'-' * 4}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-"
          f"{'-' * 7}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}")

    for m in all_metrics:
        pf_str = f"{m['profit_factor']:6.2f}" if isinstance(m["profit_factor"], (int, float)) else f"{'inf':>6}"
        print(f"  {m['portfolio_size']:4d} | {m['cagr_pct']:6.1f}% | {m['sharpe']:7.3f} | "
              f"{m['sortino']:8.3f} | {m['max_drawdown_pct']:6.1f}% | "
              f"{m['win_rate_pct']:4.1f}% | {pf_str} | {m['total_trades']:6d} | "
              f"{m['avg_hold_days']:5.1f}d")

    # Regime analysis
    print(f"\n  REGIME ANALYSIS (monthly Sharpe by SPY regime):")
    print(f"  {'Size':>4} | {'Green#':>6} | {'Green SR':>8} | {'Red#':>4} | {'Red SR':>7} | "
          f"{'Skew':>5} | {'Agnostic':>8}")
    print(f"  {'-' * 4}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 4}-+-{'-' * 7}-+-"
          f"{'-' * 5}-+-{'-' * 8}")

    for m in all_metrics:
        r = m.get("regime_analysis", {})
        agn = "PASS" if r.get("regime_agnostic", False) else "FAIL"
        print(f"  {m['portfolio_size']:4d} | {r.get('green_months', 0):6d} | "
              f"{r.get('green_sharpe_monthly', 0):8.3f} | {r.get('red_months', 0):4d} | "
              f"{r.get('red_sharpe_monthly', 0):7.3f} | {r.get('regime_skew', 0):5.3f} | "
              f"{agn:>8}")

    # Exit reasons
    print(f"\n  EXIT REASONS (across all portfolio sizes):")
    total_exits = {}
    for m in all_metrics:
        for reason, count in m.get("exit_reasons", {}).items():
            total_exits[reason] = total_exits.get(reason, 0) + count

    for reason, count in sorted(total_exits.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25}: {count:5d}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'#' * 80}")
    print(f"  WALK-FORWARD MOMENTUM BACKTEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    # 1. Build universe
    tickers = build_universe()

    # 2. Download data
    close, high, low, volume = download_data(tickers, start=DATA_START)
    spy = download_spy(start=DATA_START)

    # 3. Run backtests for each portfolio size
    all_metrics = []
    all_results = {}

    for size in PORTFOLIO_SIZES:
        print(f"\n{'─' * 60}")
        print(f"  Running backtest: Top {size} stocks")
        print(f"{'─' * 60}")

        result = run_backtest(close, high, low, spy, portfolio_size=size)
        if not result:
            print(f"  [ERROR] Backtest failed for size {size}")
            continue

        metrics = calc_metrics(result, spy)
        all_metrics.append(metrics)
        all_results[f"top_{size}"] = metrics

        print(f"  Top {size}: CAGR={metrics.get('cagr_pct', 0):.1f}%, "
              f"Sharpe={metrics.get('sharpe', 0):.3f}, "
              f"MaxDD={metrics.get('max_drawdown_pct', 0):.1f}%, "
              f"WR={metrics.get('win_rate_pct', 0):.1f}%, "
              f"Trades={metrics.get('total_trades', 0)}")

    # 4. Print summary
    print_results(all_metrics)

    # 5. Save results
    output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "lookback_days": LOOKBACK_DAYS,
            "skip_recent_days": SKIP_RECENT,
            "hold_period_days": HOLD_PERIOD,
            "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_stop_atr_mult": TRAILING_STOP_ATR_MULT,
            "momentum_breakdown_days": MOM_BREAKDOWN_DAYS,
            "cost_per_trade_pct": COST_PER_TRADE * 100,
            "data_start": DATA_START,
            "backtest_start": BACKTEST_START,
            "walk_forward": "sliding",
            "window": "expanding=NEVER",
        },
        "results": all_results,
    }

    json_path = os.path.join(output_dir, f"momentum_backtest_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[INFO] Results saved: {json_path}")

    print(f"\n{'#' * 80}")
    print(f"  BACKTEST COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    return all_results


if __name__ == "__main__":
    main()
