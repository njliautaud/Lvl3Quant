#!/usr/bin/env python3
"""
Dividend Ex-Date Trading Strategy Backtest
==========================================
Tests 6 variants of trading around ex-dividend dates using yfinance data.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Academic basis: Elton & Gruber (1970), Frank & Jagannathan (1998) — stocks
don't fully adjust down on ex-dividend dates. The drop is typically less than
the dividend amount, creating predictable price patterns.

Universe:
  ETFs:   SCHD, VYM, DVY, HDV, JEPI, JEPQ
  Stocks: T, VZ, MO, XOM, CVX, PFE, KO, PEP, JNJ, ABBV, PG, MMM

Account: $645, $0 commission, 0.02% slippage
Regime:  Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
START_DATE = "2022-01-01"
END_DATE = "2026-07-29"
PERM_ITERATIONS = 1000
TRADING_DAYS_PER_YEAR = 252

ETFS = ["SCHD", "VYM", "DVY", "HDV", "JEPI", "JEPQ"]
STOCKS = ["T", "VZ", "MO", "XOM", "CVX", "PFE", "KO", "PEP", "JNJ", "ABBV", "PG", "MMM"]
ALL_TICKERS = ETFS + STOCKS

np.random.seed(42)


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download price and dividend data for all tickers + SPY for regime."""
    print("Downloading price data...")
    tickers_needed = ALL_TICKERS + ["SPY"]

    price_data = {}
    dividend_data = {}

    for ticker in tickers_needed:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)
            if len(hist) < 50:
                print(f"  {ticker}: insufficient data ({len(hist)} rows), skipping")
                continue
            price_data[ticker] = hist[["Close", "Open", "High", "Low", "Volume"]].copy()

            if ticker != "SPY":
                divs = t.dividends
                if divs is not None and len(divs) > 0:
                    divs = divs[(divs.index >= START_DATE) & (divs.index <= END_DATE)]
                    dividend_data[ticker] = divs
                    print(f"  {ticker}: {len(hist)} bars, {len(divs)} dividends")
                else:
                    print(f"  {ticker}: {len(hist)} bars, 0 dividends")
            else:
                print(f"  SPY: {len(hist)} bars (regime benchmark)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    return price_data, dividend_data


def compute_regime(spy_prices):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_prices["Close"].rolling(200).mean()
    regime = pd.Series("bull", index=spy_prices.index)
    regime[spy_prices["Close"] < sma200] = "bear"
    return regime


def get_ex_date_info(ticker, price_data, dividend_data):
    """
    Get list of (ex_date_idx, div_amount) for each ex-dividend date.
    ex_date_idx is the index position in the price DataFrame.
    """
    if ticker not in dividend_data or len(dividend_data[ticker]) == 0:
        return []

    prices = price_data[ticker]
    divs = dividend_data[ticker]

    results = []
    for ex_date in divs.index:
        # Find closest trading day on or after ex_date
        mask = prices.index >= ex_date
        if not mask.any():
            continue
        # Get the first date >= ex_date
        matched_dates = prices.index[mask]
        if len(matched_dates) == 0:
            continue
        actual_date = matched_dates[0]
        idx = prices.index.get_loc(actual_date)
        if isinstance(idx, slice):
            idx = idx.start
        results.append((idx, float(divs.loc[ex_date])))

    return results


def compute_trailing_yield(ticker, price_data, dividend_data):
    """Compute trailing 12-month dividend yield at each date."""
    if ticker not in dividend_data or len(dividend_data[ticker]) == 0:
        return pd.Series(0.0, index=price_data[ticker].index)

    divs = dividend_data[ticker]
    prices = price_data[ticker]["Close"]
    yields = pd.Series(0.0, index=prices.index)

    for date in prices.index:
        trailing_start = date - pd.Timedelta(days=365)
        trailing_divs = divs[(divs.index >= trailing_start) & (divs.index <= date)]
        annual_div = trailing_divs.sum()
        if prices.loc[date] > 0:
            yields.loc[date] = annual_div / prices.loc[date]

    return yields


# ─── Trade Simulation ────────────────────────────────────────────────────────

def simulate_trade(prices_df, entry_idx, exit_idx, dividend_received=0.0, ticker=""):
    """
    Simulate a single long trade from entry_idx to exit_idx.
    dividend_received: dollar amount of dividend per share if holding through ex-date.
    Returns trade dict or None if indices invalid.
    """
    dates = prices_df.index
    n = len(dates)

    if entry_idx < 0 or entry_idx >= n or exit_idx < 0 or exit_idx >= n:
        return None

    entry_price = prices_df.iloc[entry_idx]["Close"]
    exit_price = prices_df.iloc[exit_idx]["Close"]

    # Apply slippage both sides
    entry_cost = entry_price * (1 + SLIPPAGE_PCT)
    exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

    # Price return + dividend income
    price_return = (exit_proceeds - entry_cost) / entry_cost
    div_return = dividend_received / entry_cost if entry_cost > 0 else 0.0
    total_return = price_return + div_return

    return {
        "ticker": ticker,
        "entry_date": str(dates[entry_idx].date()) if hasattr(dates[entry_idx], 'date') else str(dates[entry_idx])[:10],
        "exit_date": str(dates[exit_idx].date()) if hasattr(dates[exit_idx], 'date') else str(dates[exit_idx])[:10],
        "entry_price": round(float(entry_price), 2),
        "exit_price": round(float(exit_price), 2),
        "dividend_received": round(float(dividend_received), 4),
        "price_return": round(float(price_return), 6),
        "div_return": round(float(div_return), 6),
        "pct_return": round(float(total_return), 6),
        "hold_days": exit_idx - entry_idx,
    }


# ─── Strategy Variants ───────────────────────────────────────────────────────

def strategy_A_pre_ex_buy(price_data, dividend_data):
    """
    A) Pre-Ex Buy: Buy shares 5 days before ex-date, sell at close on ex-date.
    Capture the run-up into the ex-date. We do NOT hold through ex-date,
    so no dividend received.
    """
    trades = []
    for ticker in ALL_TICKERS:
        if ticker not in price_data:
            continue
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            entry_idx = ex_idx - 5
            if entry_idx < 0:
                continue
            # Sell at close on ex-date. We buy 5 days before, sell on ex-date.
            # We actually hold through ex-date if we sell AT CLOSE on ex-date,
            # meaning we owned the stock at market open on ex-date => we DO get the dividend.
            # But the spec says "sell at close on ex-date" which means we're selling
            # the shares after already being holder of record. So we get the dividend.
            # Actually: record date is before ex-date. If you own before ex-date, you get it.
            # Buying 5 days before and selling ON ex-date = you held on the record date = dividend received.
            result = simulate_trade(prices, entry_idx, ex_idx,
                                    dividend_received=div_amount, ticker=ticker)
            if result:
                result["strategy"] = "A_pre_ex_buy"
                trades.append(result)

    return trades


def strategy_B_ex_date_dip_buy(price_data, dividend_data):
    """
    B) Ex-Date Dip Buy: Buy at close on ex-date (after the drop), sell 5 days later.
    Capture recovery. No dividend (bought after ex-date).
    """
    trades = []
    for ticker in ALL_TICKERS:
        if ticker not in price_data:
            continue
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            exit_idx = ex_idx + 5
            if exit_idx >= len(prices):
                continue
            result = simulate_trade(prices, ex_idx, exit_idx,
                                    dividend_received=0.0, ticker=ticker)
            if result:
                result["strategy"] = "B_ex_date_dip_buy"
                trades.append(result)

    return trades


def strategy_C_dividend_capture(price_data, dividend_data):
    """
    C) Dividend Capture: Buy 1 day before ex-date, collect dividend,
    sell 3 days after ex-date. Net of incomplete price adjustment.
    Hold through ex-date => receive dividend.
    """
    trades = []
    for ticker in ALL_TICKERS:
        if ticker not in price_data:
            continue
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            entry_idx = ex_idx - 1
            exit_idx = ex_idx + 3
            if entry_idx < 0 or exit_idx >= len(prices):
                continue
            result = simulate_trade(prices, entry_idx, exit_idx,
                                    dividend_received=div_amount, ticker=ticker)
            if result:
                result["strategy"] = "C_dividend_capture"
                trades.append(result)

    return trades


def strategy_D_high_yield_focus(price_data, dividend_data):
    """
    D) High-Yield Focus: Only trade stocks/ETFs with dividend yield > 3%.
    Same as A (pre-ex buy 5 days before, sell on ex-date). Get dividend.
    """
    trades = []
    yield_cache = {}

    for ticker in ALL_TICKERS:
        if ticker not in price_data:
            continue
        yield_cache[ticker] = compute_trailing_yield(ticker, price_data, dividend_data)
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            # Check yield > 3% at time of trade
            ex_date = prices.index[ex_idx]
            ann_yield = yield_cache[ticker].get(ex_date, 0)
            if ann_yield < 0.03:
                continue

            entry_idx = ex_idx - 5
            if entry_idx < 0:
                continue

            result = simulate_trade(prices, entry_idx, ex_idx,
                                    dividend_received=div_amount, ticker=ticker)
            if result:
                result["strategy"] = "D_high_yield_focus"
                result["annual_yield"] = round(float(ann_yield), 4)
                trades.append(result)

    return trades


def strategy_E_etf_only(price_data, dividend_data):
    """
    E) ETF Only: Same as A but only on ETFs (SCHD, VYM, DVY, HDV).
    More liquid, predictable quarterly schedule. Buy 5d before, sell on ex-date.
    """
    etf_subset = ["SCHD", "VYM", "DVY", "HDV"]
    trades = []

    for ticker in etf_subset:
        if ticker not in price_data:
            continue
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            entry_idx = ex_idx - 5
            if entry_idx < 0:
                continue

            result = simulate_trade(prices, entry_idx, ex_idx,
                                    dividend_received=div_amount, ticker=ticker)
            if result:
                result["strategy"] = "E_etf_only"
                trades.append(result)

    return trades


def strategy_F_combo(price_data, dividend_data):
    """
    F) Combo: For each upcoming ex-date window, pick the 3 stocks with highest
    upcoming dividend yield. Buy 5 days pre-ex, sell 5 days post-ex (10-day hold).
    Maximum dividend capture window. Receive dividend.
    """
    # Collect all ex-date events with yield info
    events = []
    yield_cache = {}

    for ticker in ALL_TICKERS:
        if ticker not in price_data:
            continue
        if ticker not in yield_cache:
            yield_cache[ticker] = compute_trailing_yield(ticker, price_data, dividend_data)
        ex_dates = get_ex_date_info(ticker, price_data, dividend_data)
        prices = price_data[ticker]

        for ex_idx, div_amount in ex_dates:
            entry_idx = ex_idx - 5
            exit_idx = ex_idx + 5
            if entry_idx < 0 or exit_idx >= len(prices):
                continue
            ex_date = prices.index[ex_idx]
            ann_yield = yield_cache[ticker].get(ex_date, 0)
            # Compute per-trade yield (dividend / entry price)
            entry_price = float(prices.iloc[entry_idx]["Close"])
            trade_yield = div_amount / entry_price if entry_price > 0 else 0

            events.append({
                "ticker": ticker,
                "ex_idx": ex_idx,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "div_amount": div_amount,
                "trade_yield": trade_yield,
                "ann_yield": ann_yield,
                "ex_date": ex_date,
            })

    # Group events by approximate ex-date (within 5 trading days = same window)
    events.sort(key=lambda e: e["ex_date"])

    trades = []
    used_windows = set()

    i = 0
    while i < len(events):
        # Find cluster of events within 5 trading days
        cluster = [events[i]]
        j = i + 1
        while j < len(events) and (events[j]["ex_date"] - events[i]["ex_date"]).days <= 7:
            cluster.append(events[j])
            j += 1

        # Pick top 3 by trade_yield
        cluster.sort(key=lambda e: e["trade_yield"], reverse=True)
        top3 = cluster[:3]

        for ev in top3:
            key = (ev["ticker"], str(ev["ex_date"])[:10])
            if key in used_windows:
                continue
            used_windows.add(key)

            prices = price_data[ev["ticker"]]
            result = simulate_trade(prices, ev["entry_idx"], ev["exit_idx"],
                                    dividend_received=ev["div_amount"],
                                    ticker=ev["ticker"])
            if result:
                result["strategy"] = "F_combo"
                result["trade_yield"] = round(ev["trade_yield"], 4)
                trades.append(result)

        i = j

    return trades


# ─── Validation Framework ────────────────────────────────────────────────────

def compute_metrics(trades, regime_series):
    """Compute strategy metrics from trade list."""
    if len(trades) < 1:
        return None

    returns = np.array([t["pct_return"] for t in trades])
    n_trades = len(returns)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n_trades > 1 else 1e-9
    win_rate = np.mean(returns > 0)

    # Date range and annualization
    first_date = pd.Timestamp(trades[0]["entry_date"])
    last_date = pd.Timestamp(trades[-1]["exit_date"])
    date_range_days = (last_date - first_date).days
    years = max(date_range_days / 365.25, 0.5)
    trades_per_year = n_trades / years

    # Sharpe (annualized)
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max Drawdown on equity curve
    equity = ACCOUNT_SIZE
    peak = equity
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    final_equity = equity
    total_return = (final_equity - ACCOUNT_SIZE) / ACCOUNT_SIZE

    # Average dividend contribution
    div_returns = np.array([t.get("div_return", 0.0) for t in trades])
    avg_div_contribution = np.mean(div_returns) if len(div_returns) > 0 else 0.0

    # Regime stratification
    bull_returns = []
    bear_returns = []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if regime_series is not None:
            if regime_series.index.tz is not None:
                entry_date = entry_date.tz_localize(regime_series.index.tz)
            closest_idx = regime_series.index.get_indexer([entry_date], method="nearest")[0]
            if 0 <= closest_idx < len(regime_series):
                regime = regime_series.iloc[closest_idx]
                if regime == "bull":
                    bull_returns.append(t["pct_return"])
                else:
                    bear_returns.append(t["pct_return"])

    if len(bull_returns) > 2 and len(bear_returns) > 2:
        bull_sharpe = np.mean(bull_returns) / (np.std(bull_returns, ddof=1) + 1e-9) * np.sqrt(trades_per_year)
        bear_sharpe = np.mean(bear_returns) / (np.std(bear_returns, ddof=1) + 1e-9) * np.sqrt(trades_per_year)
        max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs
    else:
        bull_sharpe = bear_sharpe = regime_gap = None

    return {
        "n_trades": n_trades,
        "mean_return_pct": round(float(mean_ret * 100), 4),
        "std_return_pct": round(float(std_ret * 100), 4),
        "win_rate": round(float(win_rate), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "final_equity": round(float(final_equity), 2),
        "avg_div_contribution_pct": round(float(avg_div_contribution * 100), 4),
        "bull_sharpe": round(float(bull_sharpe), 4) if bull_sharpe is not None else None,
        "bear_sharpe": round(float(bear_sharpe), 4) if bear_sharpe is not None else None,
        "regime_gap": round(float(regime_gap), 4) if regime_gap is not None else None,
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
        "years": round(float(years), 2),
        "trades_per_year": round(float(trades_per_year), 1),
    }


def permutation_test(trades, n_iter=PERM_ITERATIONS):
    """
    Permutation test: shuffle the mapping between entry dates and returns.
    This breaks the relationship between ex-date timing and returns
    while preserving the return distribution.
    """
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["pct_return"] for t in trades])
    observed_mean = np.mean(returns)

    count_ge = 0
    for _ in range(n_iter):
        # Shuffle returns (equivalent to randomly assigning returns to dates)
        shuffled = np.random.permutation(returns)
        if np.mean(shuffled) >= observed_mean:
            count_ge += 1

    return count_ge / n_iter


def five_gate_validation(metrics, p_value):
    """Apply 5-gate validation."""
    if metrics is None:
        return {
            "sharpe_gt_0.5": False,
            "perm_p_lt_0.05": False,
            "regime_gap_lt_0.5": False,
            "max_dd_gt_neg50": False,
            "min_20_trades": False,
            "all_pass": False,
        }

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": (metrics["regime_gap"] < 0.5) if metrics["regime_gap"] is not None else False,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("DIVIDEND EX-DATE TRADING STRATEGY BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Account: ${ACCOUNT_SIZE}, Slippage: {SLIPPAGE_PCT*100}%, Commission: ${COMMISSION}")
    print(f"Universe: {len(ETFS)} ETFs + {len(STOCKS)} stocks = {len(ALL_TICKERS)} instruments")
    print("=" * 75)

    # Download data
    price_data, dividend_data = download_data()

    if "SPY" not in price_data:
        print("ERROR: Could not download SPY data for regime classification")
        return

    # Compute regime
    regime = compute_regime(price_data["SPY"])
    print(f"\nRegime: {(regime == 'bull').sum()} bull days, {(regime == 'bear').sum()} bear days")

    total_divs = sum(len(v) for v in dividend_data.values())
    print(f"Total dividend events across all tickers: {total_divs}")

    # Run all 6 strategy variants
    strategy_funcs = {
        "A_pre_ex_buy": ("Buy 5d before ex-date, sell on ex-date (capture run-up + dividend)", strategy_A_pre_ex_buy),
        "B_ex_date_dip_buy": ("Buy on ex-date, sell 5d later (capture recovery, no dividend)", strategy_B_ex_date_dip_buy),
        "C_dividend_capture": ("Buy 1d before ex-date, sell 3d after (capture dividend + recovery)", strategy_C_dividend_capture),
        "D_high_yield_focus": ("Same as A but only yield > 3%", strategy_D_high_yield_focus),
        "E_etf_only": ("Same as A but ETFs only (SCHD/VYM/DVY/HDV)", strategy_E_etf_only),
        "F_combo": ("Top 3 by yield per window, 5d pre to 5d post ex-date", strategy_F_combo),
    }

    results = {}

    for name, (desc, func) in strategy_funcs.items():
        print(f"\n{'─' * 60}")
        print(f"Strategy {name}: {desc}")
        print(f"{'─' * 60}")

        trades = func(price_data, dividend_data)

        if len(trades) == 0:
            print(f"  No trades generated!")
            results[name] = {
                "description": desc,
                "metrics": None,
                "gates": five_gate_validation(None, 1.0),
                "p_value": 1.0,
                "n_trades": 0,
                "verdict": "FAIL - no trades",
                "failed_gates": ["all"],
            }
            continue

        # Sort trades by entry date
        trades.sort(key=lambda x: x["entry_date"])

        metrics = compute_metrics(trades, regime)

        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
        p_value = permutation_test(trades)

        gates = five_gate_validation(metrics, p_value)
        verdict = "PASS" if gates["all_pass"] else "FAIL"
        failed_gates = [k for k, v in gates.items() if not v and k != "all_pass"]

        # Ticker breakdown
        ticker_breakdown = {}
        for t in trades:
            tk = t["ticker"]
            if tk not in ticker_breakdown:
                ticker_breakdown[tk] = {"returns": [], "div_returns": []}
            ticker_breakdown[tk]["returns"].append(t["pct_return"])
            ticker_breakdown[tk]["div_returns"].append(t.get("div_return", 0.0))

        ticker_summary = {}
        for tk, data in ticker_breakdown.items():
            rets = np.array(data["returns"])
            div_rets = np.array(data["div_returns"])
            ticker_summary[tk] = {
                "n_trades": len(rets),
                "mean_return_pct": round(float(np.mean(rets) * 100), 4),
                "win_rate": round(float(np.mean(rets > 0)), 4),
                "avg_div_pct": round(float(np.mean(div_rets) * 100), 4),
            }

        results[name] = {
            "description": desc,
            "metrics": metrics,
            "gates": gates,
            "p_value": round(float(p_value), 4),
            "n_trades": len(trades),
            "verdict": verdict,
            "failed_gates": failed_gates,
            "ticker_breakdown": ticker_summary,
            "sample_trades": trades[:5],
        }

        # Print summary
        m = metrics
        print(f"  Trades: {m['n_trades']} ({m['trades_per_year']}/yr over {m['years']} yrs)")
        print(f"  Mean Return: {m['mean_return_pct']:.4f}% (div contribution: {m['avg_div_contribution_pct']:.4f}%)")
        print(f"  Win Rate: {m['win_rate']:.1%}")
        print(f"  Sharpe: {m['sharpe']:.4f}")
        print(f"  Sortino: {m['sortino']:.4f}")
        print(f"  Profit Factor: {m['profit_factor']:.4f}")
        print(f"  Max Drawdown: {m['max_drawdown_pct']:.2f}%")
        print(f"  Total Return: {m['total_return_pct']:.2f}% (${ACCOUNT_SIZE:.0f} -> ${m['final_equity']:.2f})")
        if m['bull_sharpe'] is not None:
            print(f"  Bull Sharpe: {m['bull_sharpe']:.4f} ({m['bull_trades']} trades)")
            print(f"  Bear Sharpe: {m['bear_sharpe']:.4f} ({m['bear_trades']} trades)")
            print(f"  Regime Gap: {m['regime_gap']:.4f}")
        print(f"  Permutation p-value: {p_value:.4f}")
        print(f"  5-Gate: {verdict}")
        if failed_gates:
            print(f"  Failed gates: {', '.join(failed_gates)}")

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"{'Strategy':<25} {'Trades':>6} {'MeanRet%':>9} {'DivContr%':>9} {'Sharpe':>8} {'WR':>6} {'MaxDD':>7} {'p-val':>6} {'Verdict':>8}")
    print("-" * 95)

    for name, r in results.items():
        m = r["metrics"]
        if m:
            print(f"{name:<25} {m['n_trades']:>6} {m['mean_return_pct']:>8.4f}% "
                  f"{m['avg_div_contribution_pct']:>8.4f}% {m['sharpe']:>8.3f} "
                  f"{m['win_rate']:>5.1%} {m['max_drawdown_pct']:>6.1f}% "
                  f"{r['p_value']:>6.3f} {r['verdict']:>8}")
        else:
            print(f"{name:<25} {'0':>6} {'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>6} {'N/A':>7} {'N/A':>6} {'FAIL':>8}")

    # ─── Save Results ────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/dividend_exdate_results.json")

    serializable = {
        "metadata": {
            "strategy": "Dividend Ex-Date Trading",
            "academic_basis": "Elton & Gruber (1970), Frank & Jagannathan (1998)",
            "period": f"{START_DATE} to {END_DATE}",
            "account_size": ACCOUNT_SIZE,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
            "etfs": ETFS,
            "stocks": STOCKS,
            "n_tickers": len(ALL_TICKERS),
            "total_dividend_events": total_divs,
            "perm_iterations": PERM_ITERATIONS,
            "regime_definition": "Bull = SPY > 200-SMA, Bear = SPY < 200-SMA",
            "run_timestamp": str(dt.datetime.now()),
        },
        "variants": {},
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    for name, r in results.items():
        serializable["variants"][name] = {
            "description": r["description"],
            "metrics": r["metrics"],
            "gates": {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in r["gates"].items()},
            "p_value": r["p_value"],
            "n_trades": r["n_trades"],
            "verdict": r["verdict"],
            "failed_gates": r.get("failed_gates", []),
            "ticker_breakdown": r.get("ticker_breakdown", {}),
        }

    any_pass = any(r["verdict"] == "PASS" for r in results.values())
    serializable["overall_verdict"] = "VIABLE" if any_pass else "NO VIABLE VARIANT"

    best_name = None
    best_sharpe = -999
    for name, r in results.items():
        if r["metrics"] and r["metrics"]["sharpe"] > best_sharpe:
            best_sharpe = r["metrics"]["sharpe"]
            best_name = name
    serializable["best_variant"] = best_name
    serializable["best_sharpe"] = round(best_sharpe, 4) if best_name else None

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")
    print(f"Overall verdict: {serializable['overall_verdict']}")
    if best_name:
        print(f"Best variant: {best_name} (Sharpe: {best_sharpe:.4f})")


if __name__ == "__main__":
    main()
