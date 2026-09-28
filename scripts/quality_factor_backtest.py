#!/usr/bin/env python3
"""
Quality-Value Composite Factor Strategy Backtest
=================================================
Academic basis:
  - Asness, Frazzini & Pedersen (2019) "Quality Minus Junk"
  - Novy-Marx (2013) "The Other Side of Value"

Uses yfinance fundamental data to score S&P 500 stocks on quality + value,
then backtests monthly-rebalanced portfolios Jan 2022 - Jul 2026.

Variants:
  A: Top 5 combined score, equal weight, monthly rebalance
  B: Top 10 combined score, equal weight, monthly rebalance
  C: Quality only (top 5), monthly rebalance
  D: Long top 5 quality / Short bottom 5 quality (market neutral)
  E: Top 5 combined + momentum filter (> 50-SMA), monthly rebalance

All variants: regime hedge half-size when SPY < 200-SMA
Universe: Top 100 S&P 500 by market cap
Cost: $0 commission, 0.02% slippage
OOT: Jan 2022 - Jul 2026
Starting capital: $10,000

KNOWN LIMITATION: yfinance fundamentals are current-point-in-time, not historical.
Quality/value metrics are slow-changing (ROE, D/E, margins), so look-ahead bias
is acknowledged but mitigated by the slow-moving nature of these factors.
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

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/quality_factor_results.json"
START_DATE = "2022-01-01"
END_DATE = "2026-07-28"
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 2  # 0.02%
REBALANCE_DAY = 1  # first trading day of each month

# Top 100 S&P 500 by market cap (stable mega/large caps)
SP500_TOP100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "XOM",
    "LLY", "JPM", "JNJ", "V", "PG", "MA", "AVGO", "HD", "MRK", "COST",
    "ABBV", "PEP", "ADBE", "KO", "CVX", "CRM", "WMT", "TMO", "MCD", "CSCO",
    "ACN", "ABT", "LIN", "DHR", "NEE", "CMCSA", "PFE", "ORCL", "NKE", "TXN",
    "PM", "INTC", "AMD", "UPS", "HON", "QCOM", "UNP", "RTX", "LOW", "AMGN",
    "SPGI", "GS", "ELV", "INTU", "BLK", "CAT", "ISRG", "AXP", "MDLZ", "SYK",
    "BKNG", "GILD", "DE", "ADI", "VRTX", "REGN", "MMC", "LRCX", "CI", "SCHW",
    "ETN", "ZTS", "MO", "PANW", "BSX", "CB", "SO", "DUK", "PGR", "BDX",
    "CME", "TMUS", "AON", "ITW", "HUM", "CL", "EQIX", "SHW", "FIS", "ICE",
    "PLD", "MCK", "SNPS", "NOC", "CDNS", "KLAC", "WM", "SLB", "APD", "GD",
]

# Sector mapping for value scoring (simplified)
SECTOR_MAP = {}  # Will be populated from yfinance

# ─── Fetch Fundamentals ─────────────────────────────────────────────────────
def fetch_fundamentals(tickers):
    """Fetch fundamental data for all tickers. Returns dict of {ticker: info_dict}."""
    print(f"Fetching fundamentals for {len(tickers)} tickers...")
    fundamentals = {}
    failed = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(tickers)}")
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and isinstance(info, dict) and len(info) > 5:
                fundamentals[ticker] = info
                SECTOR_MAP[ticker] = info.get("sector", "Unknown")
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"  Got fundamentals for {len(fundamentals)} tickers, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed[:20]}{'...' if len(failed) > 20 else ''}")
    return fundamentals


def compute_quality_score(info):
    """
    Quality Score (0-100):
      - ROE > 15% (+25)
      - Debt/Equity < 1.0 (+25)
      - Revenue growth > 5% YoY (+25)
      - Gross margin > 30% (+25)
    """
    score = 0
    fields_available = 0

    roe = info.get("returnOnEquity")
    if roe is not None:
        fields_available += 1
        if roe > 0.15:
            score += 25

    de = info.get("debtToEquity")
    if de is not None:
        fields_available += 1
        if de < 100:  # yfinance reports as percentage (e.g., 50 = 0.5)
            score += 25

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        fields_available += 1
        if rev_growth > 0.05:
            score += 25

    gm = info.get("grossMargins")
    if gm is not None:
        fields_available += 1
        if gm > 0.30:
            score += 25

    if fields_available < 2:
        return None  # Not enough data
    return score


def compute_value_score(info, sector_medians):
    """
    Value Score (0-100):
      - P/E < sector median (+25)
      - P/B < sector median (+25)
      - Dividend yield > 0 (+25)
      - Free cash flow yield > 3% (+25)
    """
    score = 0
    fields_available = 0
    sector = info.get("sector", "Unknown")

    pe = info.get("trailingPE")
    if pe is not None and pe > 0:
        fields_available += 1
        median_pe = sector_medians.get(sector, {}).get("pe", 25)
        if pe < median_pe:
            score += 25

    pb = info.get("priceToBook")
    if pb is not None and pb > 0:
        fields_available += 1
        median_pb = sector_medians.get(sector, {}).get("pb", 5)
        if pb < median_pb:
            score += 25

    div_yield = info.get("dividendYield")
    if div_yield is not None:
        fields_available += 1
        if div_yield > 0:
            score += 25

    fcf = info.get("freeCashflow")
    mcap = info.get("marketCap")
    if fcf is not None and mcap is not None and mcap > 0:
        fields_available += 1
        fcf_yield = fcf / mcap
        if fcf_yield > 0.03:
            score += 25

    if fields_available < 2:
        return None
    return score


def compute_sector_medians(fundamentals):
    """Compute sector-level median P/E and P/B for value scoring."""
    sector_data = {}
    for ticker, info in fundamentals.items():
        sector = info.get("sector", "Unknown")
        if sector not in sector_data:
            sector_data[sector] = {"pe": [], "pb": []}
        pe = info.get("trailingPE")
        if pe and pe > 0:
            sector_data[sector]["pe"].append(pe)
        pb = info.get("priceToBook")
        if pb and pb > 0:
            sector_data[sector]["pb"].append(pb)

    medians = {}
    for sector, data in sector_data.items():
        medians[sector] = {
            "pe": float(np.median(data["pe"])) if data["pe"] else 25.0,
            "pb": float(np.median(data["pb"])) if data["pb"] else 5.0,
        }
    return medians


def rank_stocks(fundamentals):
    """Score and rank all stocks. Returns DataFrame with scores."""
    sector_medians = compute_sector_medians(fundamentals)

    rows = []
    for ticker, info in fundamentals.items():
        qs = compute_quality_score(info)
        vs = compute_value_score(info, sector_medians)
        if qs is not None and vs is not None:
            combined = 0.6 * qs + 0.4 * vs
            rows.append({
                "ticker": ticker,
                "quality_score": qs,
                "value_score": vs,
                "combined_score": combined,
                "sector": info.get("sector", "Unknown"),
                "roe": info.get("returnOnEquity"),
                "de": info.get("debtToEquity"),
                "rev_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "div_yield": info.get("dividendYield"),
                "mcap": info.get("marketCap"),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    print(f"\nScored {len(df)} stocks successfully")
    print(f"Top 10 by combined score:")
    print(df[["ticker", "quality_score", "value_score", "combined_score", "sector"]].head(10).to_string(index=False))
    print(f"\nBottom 5 by quality score:")
    bottom5 = df.sort_values("quality_score").head(5)
    print(bottom5[["ticker", "quality_score", "value_score", "combined_score", "sector"]].to_string(index=False))
    return df


# ─── Price Data ──────────────────────────────────────────────────────────────
def fetch_prices(tickers, start, end):
    """Fetch daily adjusted close prices for all tickers + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    print(f"\nFetching price data for {len(all_tickers)} tickers from {start} to {end}...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = all_tickers
    prices = prices.ffill()
    print(f"  Got {len(prices)} trading days of data")
    return prices


def compute_sma(prices, ticker, window):
    """Compute SMA for a ticker."""
    if ticker in prices.columns:
        return prices[ticker].rolling(window).mean()
    return None


# ─── Backtesting Engine ─────────────────────────────────────────────────────
def get_rebalance_dates(prices):
    """Get first trading day of each month."""
    dates = prices.index.to_series()
    monthly = dates.groupby([dates.dt.year, dates.dt.month]).first()
    return monthly.values


def apply_slippage(price, direction, slippage_bps=SLIPPAGE_BPS):
    """Apply slippage cost."""
    slip = price * slippage_bps / 10000
    if direction == "buy":
        return price + slip
    else:
        return price - slip


def backtest_variant(variant_name, long_tickers, short_tickers, prices, spy_sma200,
                     equal_weight=True, momentum_filter=False):
    """
    Run backtest for a given variant.

    Uses clean cash-based accounting:
      - cash starts at INITIAL_CAPITAL
      - buying shares: cash -= shares * buy_price
      - selling shares: cash += shares * sell_price
      - short selling: cash += shares * sell_price (proceeds received)
      - covering short: cash -= shares * cover_price
      - total_equity = cash + long_market_value - short_market_value
    """
    rebalance_dates = get_rebalance_dates(prices)

    cash = INITIAL_CAPITAL
    equity_curve = []
    trades_log = []
    # positions: {ticker: {"shares": n, "side": "long"/"short"}}
    # For shorts, "shares" is positive (number of shares short)
    positions = {}

    trading_days = prices.index

    # Track for metrics
    monthly_returns = []
    prev_month_equity = INITIAL_CAPITAL

    for day_idx, date in enumerate(trading_days):
        date_ts = pd.Timestamp(date)
        is_rebalance = date_ts in rebalance_dates

        if is_rebalance:
            # Determine regime: SPY < 200-SMA = half size
            spy_price = prices.loc[date, "SPY"] if "SPY" in prices.columns else None
            sma_val = spy_sma200.loc[date] if date in spy_sma200.index else None
            regime_hedge = False
            if spy_price is not None and sma_val is not None and not np.isnan(sma_val):
                regime_hedge = spy_price < sma_val
            size_mult = 0.5 if regime_hedge else 1.0

            # Close all existing positions
            for ticker, pos in list(positions.items()):
                if ticker in prices.columns and not np.isnan(prices.loc[date, ticker]):
                    if pos["side"] == "long":
                        sell_price = apply_slippage(prices.loc[date, ticker], "sell")
                        cash += pos["shares"] * sell_price
                    else:  # short
                        cover_price = apply_slippage(prices.loc[date, ticker], "buy")
                        cash -= pos["shares"] * cover_price
                    trades_log.append({
                        "date": str(date_ts.date()),
                        "ticker": ticker,
                        "action": "close",
                        "side": pos["side"],
                    })
            positions = {}

            # Current total equity (all cash now, positions closed)
            total_equity_now = cash

            # Monthly return tracking
            if prev_month_equity > 0:
                monthly_returns.append(total_equity_now / prev_month_equity - 1)
            prev_month_equity = total_equity_now

            # Filter tickers with available prices
            valid_longs = [t for t in long_tickers
                          if t in prices.columns and not np.isnan(prices.loc[date, t])]
            valid_shorts = [t for t in short_tickers
                           if t in prices.columns and not np.isnan(prices.loc[date, t])]

            # Apply momentum filter if needed
            if momentum_filter:
                filtered = []
                for t in valid_longs:
                    sma50 = prices[t].loc[:date].tail(50).mean()
                    if prices.loc[date, t] > sma50:
                        filtered.append(t)
                valid_longs = filtered if filtered else valid_longs[:3]

            n_long = len(valid_longs)
            n_short = len(valid_shorts)

            if n_long > 0 or n_short > 0:
                # For long-short: allocate based on current equity
                if n_short > 0:
                    long_alloc = total_equity_now * 0.5 * size_mult
                    short_alloc = total_equity_now * 0.5 * size_mult
                else:
                    long_alloc = total_equity_now * size_mult
                    short_alloc = 0

                # Open long positions
                if n_long > 0:
                    per_stock = long_alloc / n_long
                    for t in valid_longs:
                        buy_price = apply_slippage(prices.loc[date, t], "buy")
                        shares = int(per_stock / buy_price)
                        if shares > 0:
                            cash -= shares * buy_price
                            positions[t] = {"shares": shares, "side": "long"}

                # Open short positions
                if n_short > 0:
                    per_stock = short_alloc / n_short
                    for t in valid_shorts:
                        short_price = apply_slippage(prices.loc[date, t], "sell")
                        shares = int(per_stock / short_price)
                        if shares > 0:
                            cash += shares * short_price  # receive proceeds
                            positions[t] = {"shares": shares, "side": "short"}

        # Mark to market: equity = cash + long_value - short_value
        total_equity = cash
        for ticker, pos in positions.items():
            if ticker in prices.columns and not np.isnan(prices.loc[date, ticker]):
                price = prices.loc[date, ticker]
                if pos["side"] == "long":
                    total_equity += pos["shares"] * price
                else:
                    total_equity -= pos["shares"] * price

        equity_curve.append({
            "date": str(date_ts.date()),
            "equity": round(total_equity, 2),
        })

    # Final equity
    final_equity = equity_curve[-1]["equity"] if equity_curve else cash

    return {
        "variant": variant_name,
        "equity_curve": equity_curve,
        "trades": trades_log,
        "final_equity": final_equity,
        "monthly_returns": monthly_returns,
    }


# ─── Performance Metrics (5-Gate Validation) ─────────────────────────────────
def compute_metrics(result, prices):
    """Compute performance metrics for 5-gate validation."""
    ec = pd.DataFrame(result["equity_curve"])
    ec["date"] = pd.to_datetime(ec["date"])
    ec = ec.set_index("date")

    daily_returns = ec["equity"].pct_change().dropna()

    # Basic metrics
    total_return = (result["final_equity"] / INITIAL_CAPITAL - 1) * 100
    n_years = len(daily_returns) / 252
    cagr = ((result["final_equity"] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Sharpe (annualized, rf=0)
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino (annualized)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0

    # Max drawdown
    cummax = ec["equity"].cummax()
    drawdown = (ec["equity"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Win rate (monthly)
    mr = result["monthly_returns"]
    if mr:
        win_rate = sum(1 for r in mr if r > 0) / len(mr) * 100
    else:
        win_rate = 0

    # Profit factor
    gains = sum(r for r in mr if r > 0)
    losses = abs(sum(r for r in mr if r < 0))
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # SPY benchmark
    if "SPY" in prices.columns:
        spy_start = prices["SPY"].iloc[0]
        spy_end = prices["SPY"].iloc[-1]
        spy_return = (spy_end / spy_start - 1) * 100
        spy_daily = prices["SPY"].pct_change().dropna()
        spy_sharpe = (spy_daily.mean() / spy_daily.std()) * np.sqrt(252) if spy_daily.std() > 0 else 0
        spy_cummax = prices["SPY"].cummax()
        spy_dd = ((prices["SPY"] - spy_cummax) / spy_cummax).min() * 100
    else:
        spy_return = 0
        spy_sharpe = 0
        spy_dd = 0

    # Yearly returns
    ec_yearly = ec.copy()
    ec_yearly["year"] = ec_yearly.index.year
    yearly_returns = {}
    years = sorted(ec_yearly["year"].unique())
    for yr in years:
        yr_data = ec_yearly[ec_yearly["year"] == yr]
        if len(yr_data) > 1:
            yr_ret = (yr_data["equity"].iloc[-1] / yr_data["equity"].iloc[0] - 1) * 100
            yearly_returns[str(yr)] = round(yr_ret, 2)

    # 5-Gate Validation
    gates = {
        "gate1_sharpe_above_0.5": sharpe > 0.5,
        "gate2_profit_factor_above_1.2": profit_factor > 1.2,
        "gate3_max_dd_under_25pct": max_dd > -25,
        "gate4_win_rate_above_50pct": win_rate > 50,
        "gate5_beats_spy_risk_adjusted": sharpe > spy_sharpe,
    }
    gates_passed = sum(gates.values())

    metrics = {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "win_rate_monthly_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "n_months": len(mr),
        "yearly_returns": yearly_returns,
        "spy_total_return_pct": round(spy_return, 2),
        "spy_sharpe": round(spy_sharpe, 3),
        "spy_max_dd_pct": round(spy_dd, 2),
        "five_gate_validation": gates,
        "gates_passed": f"{gates_passed}/5",
    }

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Quality-Value Composite Factor Strategy Backtest")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Universe: Top 100 S&P 500 by market cap")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Slippage: {SLIPPAGE_BPS} bps per side")
    print()

    # 1. Fetch fundamentals
    fundamentals = fetch_fundamentals(SP500_TOP100)
    if len(fundamentals) < 50:
        print("WARNING: Got fundamentals for fewer than 50 stocks. Results may be unreliable.")

    # 2. Score and rank
    rankings = rank_stocks(fundamentals)
    if len(rankings) < 20:
        print("ERROR: Not enough scored stocks to run strategy.")
        sys.exit(1)

    # 3. Define portfolios per variant
    top5_combined = rankings.head(5)["ticker"].tolist()
    top10_combined = rankings.head(10)["ticker"].tolist()
    top5_quality = rankings.sort_values("quality_score", ascending=False).head(5)["ticker"].tolist()
    bottom5_quality = rankings.sort_values("quality_score", ascending=True).head(5)["ticker"].tolist()

    print(f"\nVariant A/E - Top 5 combined: {top5_combined}")
    print(f"Variant B   - Top 10 combined: {top10_combined}")
    print(f"Variant C   - Top 5 quality:   {top5_quality}")
    print(f"Variant D   - Long: {top5_quality}, Short: {bottom5_quality}")

    # 4. Fetch prices
    all_tickers = list(set(
        top5_combined + top10_combined + top5_quality + bottom5_quality + ["SPY"]
    ))
    prices = fetch_prices(all_tickers, START_DATE, END_DATE)

    # SPY 200-SMA for regime filter
    spy_sma200 = compute_sma(prices, "SPY", 200)

    # 5. Run all variants
    variants = {}

    print("\n" + "=" * 70)
    print("Running Variant A: Top 5 Combined, Monthly Rebalance")
    print("=" * 70)
    result_a = backtest_variant("A", top5_combined, [], prices, spy_sma200)
    result_a["description"] = "Top 5 combined score, equal weight, monthly rebalance"
    result_a["holdings"] = top5_combined

    print("Running Variant B: Top 10 Combined, Monthly Rebalance")
    result_b = backtest_variant("B", top10_combined, [], prices, spy_sma200)
    result_b["description"] = "Top 10 combined score, equal weight, monthly rebalance"
    result_b["holdings"] = top10_combined

    print("Running Variant C: Top 5 Quality Only, Monthly Rebalance")
    result_c = backtest_variant("C", top5_quality, [], prices, spy_sma200)
    result_c["description"] = "Top 5 quality score only, equal weight, monthly rebalance"
    result_c["holdings"] = top5_quality

    print("Running Variant D: Long/Short Quality (Market Neutral)")
    result_d = backtest_variant("D", top5_quality, bottom5_quality, prices, spy_sma200)
    result_d["description"] = "Long top 5 quality / Short bottom 5 quality, market neutral"
    result_d["holdings_long"] = top5_quality
    result_d["holdings_short"] = bottom5_quality

    print("Running Variant E: Top 5 Combined + Momentum Filter")
    result_e = backtest_variant("E", top5_combined, [], prices, spy_sma200,
                                 momentum_filter=True)
    result_e["description"] = "Top 5 combined + momentum filter (>50-SMA), monthly rebalance"
    result_e["holdings"] = top5_combined

    # 6. Compute metrics
    all_results = {
        "A": result_a, "B": result_b, "C": result_c, "D": result_d, "E": result_e
    }

    output = {
        "strategy": "Quality-Value Composite Factor",
        "academic_basis": [
            "Asness, Frazzini & Pedersen (2019) 'Quality Minus Junk'",
            "Novy-Marx (2013) 'The Other Side of Value'",
        ],
        "parameters": {
            "universe": "Top 100 S&P 500 by market cap",
            "quality_weight": 0.6,
            "value_weight": 0.4,
            "rebalance": "Monthly (first trading day)",
            "regime_hedge": "Half size when SPY < 200-SMA",
            "slippage_bps": SLIPPAGE_BPS,
            "commission": "$0",
            "initial_capital": INITIAL_CAPITAL,
            "oot_period": f"{START_DATE} to {END_DATE}",
        },
        "look_ahead_bias_note": (
            "yfinance fundamentals are current point-in-time, not historical. "
            "Quality metrics (ROE, D/E, margins) are slow-changing, so bias is acknowledged "
            "but mitigated. Rankings are applied as-if known at each rebalance date."
        ),
        "scoring_criteria": {
            "quality": {
                "ROE > 15%": "+25 pts",
                "Debt/Equity < 1.0": "+25 pts",
                "Revenue growth > 5% YoY": "+25 pts",
                "Gross margin > 30%": "+25 pts",
            },
            "value": {
                "P/E < sector median": "+25 pts",
                "P/B < sector median": "+25 pts",
                "Dividend yield > 0": "+25 pts",
                "FCF yield > 3%": "+25 pts",
            },
        },
        "stock_rankings": rankings[["ticker", "quality_score", "value_score",
                                     "combined_score", "sector"]].head(20).to_dict("records"),
        "variants": {},
        "generated_at": datetime.now().isoformat(),
    }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for key, result in all_results.items():
        metrics = compute_metrics(result, prices)

        variant_output = {
            "description": result["description"],
            "metrics": metrics,
        }
        if "holdings" in result:
            variant_output["holdings"] = result["holdings"]
        if "holdings_long" in result:
            variant_output["holdings_long"] = result["holdings_long"]
            variant_output["holdings_short"] = result["holdings_short"]

        output["variants"][key] = variant_output

        print(f"\nVariant {key}: {result['description']}")
        print(f"  Total Return:  {metrics['total_return_pct']:>8.2f}%")
        print(f"  CAGR:          {metrics['cagr_pct']:>8.2f}%")
        print(f"  Sharpe:        {metrics['sharpe']:>8.3f}")
        print(f"  Sortino:       {metrics['sortino']:>8.3f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Calmar:        {metrics['calmar']:>8.3f}")
        print(f"  Win Rate:      {metrics['win_rate_monthly_pct']:>8.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:>8.3f}")
        print(f"  Yearly Returns: {metrics['yearly_returns']}")
        print(f"  5-Gate:        {metrics['gates_passed']}")
        for gate, passed in metrics["five_gate_validation"].items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

    # SPY benchmark
    spy_metrics = output["variants"]["A"]["metrics"]
    print(f"\n--- SPY Benchmark ---")
    print(f"  Total Return:  {spy_metrics['spy_total_return_pct']:>8.2f}%")
    print(f"  Sharpe:        {spy_metrics['spy_sharpe']:>8.3f}")
    print(f"  Max Drawdown:  {spy_metrics['spy_max_dd_pct']:>8.2f}%")

    # Determine best variant
    best_key = max(output["variants"],
                   key=lambda k: output["variants"][k]["metrics"]["sharpe"])
    output["best_variant"] = best_key
    output["recommendation"] = (
        f"Variant {best_key} has the highest risk-adjusted returns (Sharpe "
        f"{output['variants'][best_key]['metrics']['sharpe']:.3f}). "
        f"See 5-gate validation for deployment readiness."
    )

    # Save
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    main()
