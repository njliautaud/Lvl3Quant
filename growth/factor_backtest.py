#!/usr/bin/env python3
"""
Walk-Forward Multi-Factor Backtest (Quality + Momentum + Value)
=================================================================
Backtests a composite factor strategy with:
  - MOMENTUM (40%): 12-1 month price momentum
  - QUALITY  (35%): low 60-day realized volatility (stable business = quality)
  - VALUE    (25%): earnings yield proxy via trailing return inversion
                    (cheap stocks tend to lag — price-only proxy, no API calls)

All factors are PRICE-ONLY to avoid yfinance rate limits.

Walk-forward design:
  - SLIDING window (HC #0) — never expanding
  - Monthly rebalance (21 trading days)
  - Equal-weight, 0.1% transaction cost
  - Dynamic exits: 8% stop-loss, 2x ATR trailing stop, momentum breakdown
  - Period: 2019-01-01 to present

Compares:
  - Multi-factor Top 10/15/20
  - Momentum-only Top 10 (baseline)
  - SPY buy-and-hold

Usage:
  python3 growth/factor_backtest.py
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from universe import build_universe


# ── Constants ────────────────────────────────────────────────────────────────

LOOKBACK_DAYS = 252       # 12 months for momentum
SKIP_RECENT = 21          # skip most recent month (12-1 momentum)
HOLD_PERIOD = 21          # monthly rebalance
ATR_PERIOD = 20
STOP_LOSS_PCT = 0.08
TRAILING_STOP_ATR_MULT = 2.0
MOM_BREAKDOWN_DAYS = 10
COST_PER_TRADE = 0.001    # 0.1% round-trip
VOL_LOOKBACK = 60         # 60-day realized vol for quality factor
VALUE_LOOKBACK = 126      # 6-month trailing return for value proxy

# Factor weights (must sum to 1.0)
W_MOMENTUM = 0.40
W_QUALITY = 0.35
W_VALUE = 0.25

PORTFOLIO_SIZES = [10, 15, 20]

DATA_START = "2018-01-01"
BACKTEST_START = "2019-01-01"


# ── Data Download (reuses momentum_backtest pattern) ─────────────────────────

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
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    return spy


# ── Factor Calculations ─────────────────────────────────────────────────────

def calc_momentum_12_1(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """12-1 month momentum at a specific index position. No lookahead."""
    start_idx = max(0, as_of_idx - LOOKBACK_DAYS)
    end_idx = as_of_idx - SKIP_RECENT
    if end_idx <= start_idx:
        return pd.Series(dtype=float)
    price_start = prices.iloc[start_idx]
    price_end = prices.iloc[end_idx]
    momentum = (price_end / price_start) - 1.0
    return momentum.dropna()


def calc_quality_lowvol(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    Quality proxy: 60-day realized volatility (annualized).
    LOW vol = HIGH quality. We return negative vol so higher = better quality.
    No lookahead: uses data up to as_of_idx only.
    """
    start_idx = max(0, as_of_idx - VOL_LOOKBACK)
    end_idx = as_of_idx + 1
    if end_idx - start_idx < 20:
        return pd.Series(dtype=float)
    window = prices.iloc[start_idx:end_idx]
    daily_ret = window.pct_change().dropna()
    if len(daily_ret) < 20:
        return pd.Series(dtype=float)
    vol = daily_ret.std() * np.sqrt(252)  # annualized vol
    # Invert: low vol = high quality score
    quality = -vol
    return quality.dropna()


def calc_value_proxy(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    Value proxy: inverse of trailing 6-month return.
    Stocks that have LAGGED (lower trailing return) are cheaper (value).
    This is a price-only proxy for earnings yield — avoids API calls.
    No lookahead.
    """
    start_idx = max(0, as_of_idx - VALUE_LOOKBACK)
    if as_of_idx - start_idx < 60:
        return pd.Series(dtype=float)
    price_start = prices.iloc[start_idx]
    price_end = prices.iloc[as_of_idx]
    trailing_ret = (price_end / price_start) - 1.0
    # Invert: lower trailing return = higher value score
    value = -trailing_ret
    return value.dropna()


def calc_composite_score(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    Composite factor score = weighted average of percentile ranks.
    40% momentum + 35% quality (low-vol) + 25% value (return inversion)
    Returns composite score (higher = better).
    """
    momentum = calc_momentum_12_1(prices, as_of_idx)
    quality = calc_quality_lowvol(prices, as_of_idx)
    value = calc_value_proxy(prices, as_of_idx)

    # Intersect tickers that have all three scores
    common = momentum.index.intersection(quality.index).intersection(value.index)
    if len(common) < 20:
        # Fallback: use momentum only if not enough tickers have all factors
        return momentum

    momentum = momentum.loc[common]
    quality = quality.loc[common]
    value = value.loc[common]

    # Convert to percentile ranks (0 to 1, higher = better)
    n = len(common)
    mom_rank = momentum.rank(pct=True)
    qual_rank = quality.rank(pct=True)
    val_rank = value.rank(pct=True)

    composite = (W_MOMENTUM * mom_rank +
                 W_QUALITY * qual_rank +
                 W_VALUE * val_rank)

    return composite


# ── Position Tracking ────────────────────────────────────────────────────────

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
        if current_price > self.trailing_high:
            self.trailing_high = current_price
        if current_price <= self.entry_price * (1 - STOP_LOSS_PCT):
            return True, "stop_loss"
        if not np.isnan(atr_value) and atr_value > 0:
            trail_level = self.trailing_high - TRAILING_STOP_ATR_MULT * atr_value
            if current_price <= trail_level:
                return True, "trailing_stop"
        if not np.isnan(mom_10d) and mom_10d < 0:
            return True, "momentum_breakdown"
        return False, None

    def pnl(self, exit_price):
        gross_proceeds = exit_price * self.shares * (1 - COST_PER_TRADE)
        return gross_proceeds - self.cost_basis


# ── Walk-Forward Backtest Engine ─────────────────────────────────────────────

def run_backtest(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 spy: pd.Series, portfolio_size: int,
                 scoring_fn=None, label: str = "factor") -> dict:
    """
    Run sliding walk-forward backtest.
    scoring_fn: callable(prices, as_of_idx) -> pd.Series of scores (higher = better).
    """
    if scoring_fn is None:
        scoring_fn = calc_composite_score

    dates = close.index
    n_dates = len(dates)

    backtest_start = pd.Timestamp(BACKTEST_START)
    first_valid_idx = 0
    for i in range(n_dates):
        if dates[i] >= backtest_start and i >= LOOKBACK_DAYS:
            first_valid_idx = i
            break

    if first_valid_idx == 0:
        print(f"[ERROR] Not enough data to start backtest at {BACKTEST_START}")
        return {}

    cash = 1_000_000.0
    initial_capital = cash
    positions = {}
    equity_curve = []
    trade_log = []
    exit_reasons = {"stop_loss": 0, "trailing_stop": 0, "momentum_breakdown": 0, "rebalance": 0}

    rebalance_indices = list(range(first_valid_idx, n_dates, HOLD_PERIOD))
    current_rebalance_ptr = 0
    next_rebalance_idx = rebalance_indices[0] if rebalance_indices else n_dates

    for day_idx in range(first_valid_idx, n_dates):
        current_date = dates[day_idx]

        # ── Daily exit checks ──
        tickers_to_exit = []
        for ticker, pos in positions.items():
            try:
                current_price = close[ticker].iloc[day_idx]
                if np.isnan(current_price):
                    tickers_to_exit.append((ticker, "missing_data"))
                    continue

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

        for ticker, reason in tickers_to_exit:
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
            # Close all positions at rebalance
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

            # Score and rank stocks
            scores = scoring_fn(close, day_idx)

            # Filter valid
            valid_tickers = []
            for ticker in scores.index:
                try:
                    price = close[ticker].iloc[day_idx]
                    if not np.isnan(price) and price > 5.0 and not np.isnan(scores[ticker]):
                        valid_tickers.append(ticker)
                except Exception:
                    pass

            scores = scores[valid_tickers].sort_values(ascending=False)
            selected = scores.head(portfolio_size).index.tolist()

            if len(selected) > 0:
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

            current_rebalance_ptr += 1
            if current_rebalance_ptr < len(rebalance_indices):
                next_rebalance_idx = rebalance_indices[current_rebalance_ptr]
            else:
                next_rebalance_idx = n_dates

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

    # Close remaining positions
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
        "label": label,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "exit_reasons": exit_reasons,
        "initial_capital": initial_capital,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(result: dict, spy: pd.Series) -> dict:
    """Calculate risk-adjusted performance metrics."""
    eq = pd.DataFrame(result["equity_curve"])
    if eq.empty:
        return {}

    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")
    eq["return"] = eq["equity"].pct_change()
    daily_returns = eq["return"].dropna()

    if len(daily_returns) < 30:
        return {"error": "Not enough data"}

    total_return = eq["equity"].iloc[-1] / result["initial_capital"] - 1
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

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

    # SPY benchmark
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

    # Regime analysis
    monthly_eq = eq["equity"].resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna()
    spy_monthly = spy.resample("ME").last()
    spy_monthly_ret = spy_monthly.pct_change().dropna()

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
        "label": result.get("label", "unknown"),
        "portfolio_size": result["portfolio_size"],
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
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
    """Print formatted multi-factor backtest results."""
    print(f"\n{'=' * 90}")
    print(f"  MULTI-FACTOR BACKTEST RESULTS — Walk-Forward Sliding Window")
    print(f"  Factors: Momentum (40%) + Quality/LowVol (35%) + Value/RetInv (25%)")
    print(f"{'=' * 90}")

    if not all_metrics:
        print("[ERROR] No results to display.")
        return

    period = all_metrics[0].get("backtest_period", {})
    print(f"\n  Period: {period.get('start', '?')} to {period.get('end', '?')} "
          f"({period.get('years', '?')} years, {period.get('trading_days', '?')} trading days)")

    spy_b = all_metrics[0].get("benchmark_spy", {})
    print(f"\n  SPY Buy-and-Hold: CAGR {spy_b.get('cagr_pct', 0):.1f}% | "
          f"Sharpe {spy_b.get('sharpe', 0):.3f} | "
          f"MaxDD {spy_b.get('max_drawdown_pct', 0):.1f}%")

    print(f"\n  {'Strategy':<22} | {'CAGR':>7} | {'Sharpe':>7} | {'Sortino':>8} | "
          f"{'MaxDD':>7} | {'Calmar':>7} | {'WR':>5} | {'PF':>6} | {'Trades':>6}")
    print(f"  {'-' * 22}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-"
          f"{'-' * 7}-+-{'-' * 7}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 6}")

    for m in all_metrics:
        pf_str = f"{m['profit_factor']:6.2f}" if isinstance(m["profit_factor"], (int, float)) else f"{'inf':>6}"
        label = f"{m['label']} Top{m['portfolio_size']}"
        print(f"  {label:<22} | {m['cagr_pct']:6.1f}% | {m['sharpe']:7.3f} | "
              f"{m['sortino']:8.3f} | {m['max_drawdown_pct']:6.1f}% | "
              f"{m['calmar']:7.3f} | {m['win_rate_pct']:4.1f}% | {pf_str} | "
              f"{m['total_trades']:6d}")

    # Regime analysis
    print(f"\n  REGIME ANALYSIS (monthly Sharpe by SPY regime):")
    print(f"  {'Strategy':<22} | {'Green#':>6} | {'Green SR':>8} | {'Red#':>4} | {'Red SR':>7} | "
          f"{'Skew':>5} | {'Agnostic':>8}")
    print(f"  {'-' * 22}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 4}-+-{'-' * 7}-+-"
          f"{'-' * 5}-+-{'-' * 8}")

    for m in all_metrics:
        r = m.get("regime_analysis", {})
        agn = "PASS" if r.get("regime_agnostic", False) else "FAIL"
        label = f"{m['label']} Top{m['portfolio_size']}"
        print(f"  {label:<22} | {r.get('green_months', 0):6d} | "
              f"{r.get('green_sharpe_monthly', 0):8.3f} | {r.get('red_months', 0):4d} | "
              f"{r.get('red_sharpe_monthly', 0):7.3f} | {r.get('regime_skew', 0):5.3f} | "
              f"{agn:>8}")

    # Factor comparison summary
    print(f"\n  FACTOR VALUE-ADD (multi-factor vs momentum-only, same portfolio size):")
    factor_results = [m for m in all_metrics if m["label"] == "factor"]
    mom_results = {m["portfolio_size"]: m for m in all_metrics if m["label"] == "momentum"}
    for f in factor_results:
        sz = f["portfolio_size"]
        if sz in mom_results:
            m = mom_results[sz]
            delta_sharpe = f["sharpe"] - m["sharpe"]
            delta_cagr = f["cagr_pct"] - m["cagr_pct"]
            delta_dd = f["max_drawdown_pct"] - m["max_drawdown_pct"]
            print(f"  Top{sz}: Sharpe {delta_sharpe:+.3f} | CAGR {delta_cagr:+.1f}pp | "
                  f"MaxDD {delta_dd:+.1f}pp "
                  f"{'[IMPROVED]' if delta_sharpe > 0 else '[WORSE]'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def calc_mom_quality_score(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    Momentum + Quality only (no value factor).
    60% momentum + 40% quality (low-vol).
    Academic evidence: momentum + low-vol is the strongest 2-factor combo.
    """
    momentum = calc_momentum_12_1(prices, as_of_idx)
    quality = calc_quality_lowvol(prices, as_of_idx)

    common = momentum.index.intersection(quality.index)
    if len(common) < 20:
        return momentum

    momentum = momentum.loc[common]
    quality = quality.loc[common]

    mom_rank = momentum.rank(pct=True)
    qual_rank = quality.rank(pct=True)

    return 0.60 * mom_rank + 0.40 * qual_rank


def calc_quality_filtered_momentum(prices: pd.DataFrame, as_of_idx: int) -> pd.Series:
    """
    Quality-filtered momentum: rank by momentum, but EXCLUDE bottom quartile
    by quality (high-vol stocks). This preserves momentum alpha while removing
    the most dangerous momentum names (high-vol momentum = crash risk).
    """
    momentum = calc_momentum_12_1(prices, as_of_idx)
    quality = calc_quality_lowvol(prices, as_of_idx)

    common = momentum.index.intersection(quality.index)
    if len(common) < 20:
        return momentum

    momentum = momentum.loc[common]
    quality = quality.loc[common]

    # Exclude bottom 25% quality (highest volatility)
    qual_rank = quality.rank(pct=True)
    quality_pass = qual_rank[qual_rank >= 0.25].index

    return momentum.loc[quality_pass]


def main():
    print(f"\n{'#' * 90}")
    print(f"  WALK-FORWARD MULTI-FACTOR BACKTEST (Quality + Momentum + Value)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 90}\n")

    # 1. Build universe
    tickers = build_universe()

    # 2. Download data (single download, used by all strategies)
    close, high, low, volume = download_data(tickers, start=DATA_START)
    spy = download_spy(start=DATA_START)

    all_metrics = []
    all_results = {}

    # 3. Run MOMENTUM-ONLY baseline (Top 10 for comparison)
    print(f"\n{'=' * 60}")
    print(f"  MOMENTUM-ONLY BASELINE (Top 10)")
    print(f"{'=' * 60}")

    def momentum_only_scorer(prices, as_of_idx):
        return calc_momentum_12_1(prices, as_of_idx)

    mom_result = run_backtest(close, high, low, spy, portfolio_size=10,
                              scoring_fn=momentum_only_scorer, label="momentum")
    if mom_result:
        mom_metrics = calc_metrics(mom_result, spy)
        all_metrics.append(mom_metrics)
        all_results["momentum_top10"] = mom_metrics
        print(f"  Momentum Top10: CAGR={mom_metrics.get('cagr_pct', 0):.1f}%, "
              f"Sharpe={mom_metrics.get('sharpe', 0):.3f}, "
              f"MaxDD={mom_metrics.get('max_drawdown_pct', 0):.1f}%")

    # 4. Run strategies across portfolio sizes
    strategies = [
        ("factor", calc_composite_score,
         f"3-Factor (Mom{W_MOMENTUM:.0%}+Qual{W_QUALITY:.0%}+Val{W_VALUE:.0%})"),
        ("mom+qual", calc_mom_quality_score,
         "Mom+Quality (60/40)"),
        ("qual_filter", calc_quality_filtered_momentum,
         "Quality-Filtered Mom (excl bottom 25% vol)"),
    ]

    for strat_label, scoring_fn, description in strategies:
        for size in PORTFOLIO_SIZES:
            print(f"\n{'─' * 60}")
            print(f"  {description}: Top {size}")
            print(f"{'─' * 60}")

            result = run_backtest(close, high, low, spy, portfolio_size=size,
                                  scoring_fn=scoring_fn, label=strat_label)
            if not result:
                print(f"  [ERROR] Backtest failed for {strat_label} size {size}")
                continue

            metrics = calc_metrics(result, spy)
            all_metrics.append(metrics)
            all_results[f"{strat_label}_top_{size}"] = metrics

            print(f"  {strat_label} Top {size}: CAGR={metrics.get('cagr_pct', 0):.1f}%, "
                  f"Sharpe={metrics.get('sharpe', 0):.3f}, "
                  f"Sortino={metrics.get('sortino', 0):.3f}, "
                  f"MaxDD={metrics.get('max_drawdown_pct', 0):.1f}%, "
                  f"Calmar={metrics.get('calmar', 0):.3f}, "
                  f"WR={metrics.get('win_rate_pct', 0):.1f}%, "
                  f"Trades={metrics.get('total_trades', 0)}")

    # 5. Print comparison summary
    print_results(all_metrics)

    # 6. Save results
    output_dir = str(Path(__file__).parent / "output")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    output = {
        "run_date": datetime.now().isoformat(),
        "description": "Multi-factor backtest: Momentum(40%) + Quality/LowVol(35%) + Value/RetInversion(25%)",
        "config": {
            "lookback_days": LOOKBACK_DAYS,
            "skip_recent_days": SKIP_RECENT,
            "hold_period_days": HOLD_PERIOD,
            "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_stop_atr_mult": TRAILING_STOP_ATR_MULT,
            "momentum_breakdown_days": MOM_BREAKDOWN_DAYS,
            "cost_per_trade_pct": COST_PER_TRADE * 100,
            "vol_lookback_days": VOL_LOOKBACK,
            "value_lookback_days": VALUE_LOOKBACK,
            "factor_weights": {
                "momentum": W_MOMENTUM,
                "quality_lowvol": W_QUALITY,
                "value_retinv": W_VALUE,
            },
            "data_start": DATA_START,
            "backtest_start": BACKTEST_START,
            "walk_forward": "sliding",
            "window": "expanding=NEVER",
        },
        "results": all_results,
    }

    json_path = os.path.join(output_dir, f"factor_backtest_{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[INFO] Results saved: {json_path}")

    print(f"\n{'#' * 90}")
    print(f"  BACKTEST COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 90}\n")

    return all_results


if __name__ == "__main__":
    main()
