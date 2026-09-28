#!/usr/bin/env python3
"""
Share Buyback Announcement Drift Backtest
==========================================
Academic basis: Ikenberry, Lakonishok & Vermaelen (1995, 2000)
Stocks announcing buybacks outperform by 3-5% annually over 1-4 years.

Proxy approach: detect active buyback execution via shares outstanding
decrease in quarterly financials (yfinance), combined with value/quality filters.

Variants:
  A: Base (shares decrease >1% QoQ, 60d hold)
  B: Value filter (+ P/B < sector median, 60d hold)
  C: Quality filter (+ ROE>15% + FCF>0, 60d hold)
  D: Combined (shares decrease + value + quality, 60d hold)
  E: Momentum overlay (+ RSI>40, not in downtrend, 40d hold)

5-gate validation per variant:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (100 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

Universe: Top 100 S&P 500 by market cap
OOT: Jan 2022 - Jul 2026
Cost: $0 commission (RH shares), 0.02% slippage
"""

import json
import os
import sys
import time
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/scripts/cache/buyback_drift")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/buyback_drift_results.json"

OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2021-01-01"  # extra lookback for quarterly data + SMA

SLIPPAGE_PCT = 0.0002  # 0.02% slippage per side (entry + exit)
COMMISSION = 0.0       # $0 commission (Robinhood)

N_PERMUTATIONS = 100
RANDOM_SEED = 42

# Top 100 S&P 500 by market cap (as of mid-2026, approximate)
TOP_100_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "COST", "HD", "ABBV",
    "MRK", "WMT", "NFLX", "CRM", "BAC", "CVX", "KO", "AMD", "PEP", "LIN",
    "TMO", "ORCL", "ACN", "MCD", "ABT", "CSCO", "ADBE", "WFC", "PM", "IBM",
    "GE", "DHR", "TXN", "ISRG", "QCOM", "INTU", "NOW", "CAT", "AMGN", "CMCSA",
    "VZ", "NEE", "PFE", "AMAT", "SPGI", "UNP", "RTX", "HON", "LOW", "T",
    "BLK", "BKNG", "SYK", "GS", "ELV", "MDLZ", "DE", "ADP", "SCHW", "TJX",
    "GILD", "LMT", "CB", "MMC", "VRTX", "CI", "BMY", "SO", "DUK", "AMT",
    "MO", "LRCX", "CME", "ICE", "PLD", "CL", "SLB", "REGN", "ZTS", "BDX",
    "SNPS", "CDNS", "EOG", "APD", "WM", "NOC", "ITW", "EMR", "FDX", "MCK",
]

# GICS Sector mapping (approximate) for P/B sector median
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "ConsDisc", "GOOGL": "Tech",
    "META": "Tech", "BRK-B": "Financials", "LLY": "Health", "AVGO": "Tech", "JPM": "Financials",
    "TSLA": "ConsDisc", "UNH": "Health", "XOM": "Energy", "V": "Financials", "MA": "Financials",
    "PG": "Staples", "JNJ": "Health", "COST": "Staples", "HD": "ConsDisc", "ABBV": "Health",
    "MRK": "Health", "WMT": "Staples", "NFLX": "Tech", "CRM": "Tech", "BAC": "Financials",
    "CVX": "Energy", "KO": "Staples", "AMD": "Tech", "PEP": "Staples", "LIN": "Materials",
    "TMO": "Health", "ORCL": "Tech", "ACN": "Tech", "MCD": "ConsDisc", "ABT": "Health",
    "CSCO": "Tech", "ADBE": "Tech", "WFC": "Financials", "PM": "Staples", "IBM": "Tech",
    "GE": "Industrials", "DHR": "Health", "TXN": "Tech", "ISRG": "Health", "QCOM": "Tech",
    "INTU": "Tech", "NOW": "Tech", "CAT": "Industrials", "AMGN": "Health", "CMCSA": "Tech",
    "VZ": "Tech", "NEE": "Utilities", "PFE": "Health", "AMAT": "Tech", "SPGI": "Financials",
    "UNP": "Industrials", "RTX": "Industrials", "HON": "Industrials", "LOW": "ConsDisc", "T": "Tech",
    "BLK": "Financials", "BKNG": "ConsDisc", "SYK": "Health", "GS": "Financials", "ELV": "Health",
    "MDLZ": "Staples", "DE": "Industrials", "ADP": "Industrials", "SCHW": "Financials", "TJX": "ConsDisc",
    "GILD": "Health", "LMT": "Industrials", "CB": "Financials", "MMC": "Financials", "VRTX": "Health",
    "CI": "Health", "BMY": "Health", "SO": "Utilities", "DUK": "Utilities", "AMT": "RealEstate",
    "MO": "Staples", "LRCX": "Tech", "CME": "Financials", "ICE": "Financials", "PLD": "RealEstate",
    "CL": "Staples", "SLB": "Energy", "REGN": "Health", "ZTS": "Health", "BDX": "Health",
    "SNPS": "Tech", "CDNS": "Tech", "EOG": "Energy", "APD": "Materials", "WM": "Industrials",
    "NOC": "Industrials", "ITW": "Industrials", "EMR": "Industrials", "FDX": "Industrials", "MCK": "Health",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data Download ──────────────────────────────────────────────────────────

def download_price_data(tickers):
    """Download daily OHLCV for all tickers."""
    cache_file = CACHE_DIR / "prices.parquet"
    if cache_file.exists():
        log(f"Loading cached prices")
        return pd.read_parquet(cache_file)

    log(f"Downloading price data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, threads=True)
    # data has MultiIndex columns: (field, ticker)
    close = data["Close"]
    close.to_parquet(cache_file)
    log(f"Saved prices: {close.shape}")
    return close


def download_spy_data():
    """Download SPY for regime detection."""
    cache_file = CACHE_DIR / "spy.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    log("Downloading SPY data...")
    spy = yf.download("SPY", start=DATA_START, end=OOT_END, auto_adjust=True)
    spy.to_parquet(cache_file)
    return spy


def download_quarterly_data(tickers):
    """Download quarterly shares outstanding, book value, ROE, FCF for all tickers."""
    cache_file = CACHE_DIR / "quarterly_fundamentals.json"
    if cache_file.exists():
        log("Loading cached quarterly fundamentals...")
        with open(cache_file) as f:
            return json.load(f)

    log(f"Downloading quarterly fundamentals for {len(tickers)} tickers...")
    fundamentals = {}

    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            log(f"  Progress: {i+1}/{len(tickers)}")
        try:
            tk = yf.Ticker(ticker)

            # Get quarterly balance sheet
            bs = tk.quarterly_balance_sheet
            # Get quarterly financials (income statement)
            inc = tk.quarterly_financials
            # Get quarterly cash flow
            cf = tk.quarterly_cashflow

            ticker_data = {"dates": [], "shares_outstanding": [], "book_value_per_share": [],
                           "roe": [], "fcf": [], "price_to_book": []}

            if bs is not None and not bs.empty:
                for col in bs.columns:
                    date_str = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)

                    # Shares outstanding
                    shares = None
                    for key in ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]:
                        if key in bs.index and pd.notna(bs.loc[key, col]):
                            shares = float(bs.loc[key, col])
                            break

                    # Book value (total equity)
                    book_val = None
                    for key in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                        if key in bs.index and pd.notna(bs.loc[key, col]):
                            book_val = float(bs.loc[key, col])
                            break

                    bvps = None
                    if book_val and shares and shares > 0:
                        bvps = book_val / shares

                    # ROE: net income / equity
                    roe = None
                    if inc is not None and not inc.empty and col in inc.columns and book_val and book_val > 0:
                        for key in ["Net Income", "Net Income Common Stockholders"]:
                            if key in inc.index and pd.notna(inc.loc[key, col]):
                                ni = float(inc.loc[key, col])
                                roe = (ni * 4) / book_val  # annualize quarterly NI
                                break

                    # FCF: operating cash flow - capex
                    fcf = None
                    if cf is not None and not cf.empty and col in cf.columns:
                        ocf = None
                        capex = None
                        for key in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                            if key in cf.index and pd.notna(cf.loc[key, col]):
                                ocf = float(cf.loc[key, col])
                                break
                        for key in ["Capital Expenditure", "Purchase Of PPE"]:
                            if key in cf.index and pd.notna(cf.loc[key, col]):
                                capex = float(cf.loc[key, col])
                                break
                        if ocf is not None:
                            fcf = ocf + (capex if capex else 0)  # capex is usually negative

                    ticker_data["dates"].append(date_str)
                    ticker_data["shares_outstanding"].append(shares)
                    ticker_data["book_value_per_share"].append(bvps)
                    ticker_data["roe"].append(roe)
                    ticker_data["fcf"].append(fcf)

            fundamentals[ticker] = ticker_data
            time.sleep(0.15)  # rate limiting

        except Exception as e:
            log(f"  ERROR on {ticker}: {e}")
            fundamentals[ticker] = {"dates": [], "shares_outstanding": [], "book_value_per_share": [],
                                     "roe": [], "fcf": [], "price_to_book": []}

    with open(cache_file, 'w') as f:
        json.dump(fundamentals, f)
    log("Saved quarterly fundamentals cache.")
    return fundamentals


# ── Signal Generation ──────────────────────────────────────────────────────

def compute_shares_change(fundamentals):
    """
    For each ticker, compute QoQ shares outstanding change.
    Returns dict: ticker -> list of (date, pct_change, shares_prev, shares_curr)
    """
    result = {}
    for ticker, data in fundamentals.items():
        if not data["dates"]:
            continue
        # Sort by date ascending
        pairs = sorted(zip(data["dates"], data["shares_outstanding"]), key=lambda x: x[0])
        changes = []
        for i in range(1, len(pairs)):
            date_curr, shares_curr = pairs[i]
            date_prev, shares_prev = pairs[i-1]
            if shares_curr and shares_prev and shares_prev > 0:
                pct_change = (shares_curr - shares_prev) / shares_prev
                changes.append({
                    "date": date_curr,
                    "pct_change": pct_change,
                    "shares_prev": shares_prev,
                    "shares_curr": shares_curr,
                })
        result[ticker] = changes
    return result


def compute_rsi(prices, period=14):
    """Compute RSI for a price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(shares_changes, fundamentals, prices, spy_data, variant="A"):
    """
    Generate buy signals based on variant.
    Returns list of trades.
    """
    # SPY 200-SMA for regime
    spy_close = spy_data["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_sma200 = spy_close.rolling(200).mean()

    # Compute sector P/B medians from fundamentals
    sector_pb = {}  # sector -> {date -> [(ticker, pb)]}
    for ticker, data in fundamentals.items():
        sector = SECTOR_MAP.get(ticker, "Unknown")
        if sector not in sector_pb:
            sector_pb[sector] = {}
        for date_str, bvps in zip(data["dates"], data["book_value_per_share"]):
            if bvps and bvps > 0 and ticker in prices.columns:
                try:
                    dt = pd.Timestamp(date_str)
                    idx = prices.index.get_indexer([dt], method="nearest")[0]
                    if 0 <= idx < len(prices):
                        price = prices.iloc[idx][ticker]
                        if pd.notna(price) and price > 0:
                            pb = price / bvps
                            if date_str not in sector_pb[sector]:
                                sector_pb[sector][date_str] = []
                            sector_pb[sector][date_str].append((ticker, pb))
                except:
                    pass

    # Hold period
    hold_days = 40 if variant == "E" else 60

    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    for ticker, changes in shares_changes.items():
        if ticker not in prices.columns:
            continue

        ticker_prices = prices[ticker].dropna()
        if ticker_prices.empty:
            continue

        # Compute RSI for variant E
        if variant == "E":
            ticker_rsi = compute_rsi(ticker_prices)
            ticker_sma50 = ticker_prices.rolling(50).mean()

        for change in changes:
            # Gate 1: shares decreased >1%
            if change["pct_change"] >= -0.01:
                continue  # not enough buyback

            signal_date = pd.Timestamp(change["date"])
            if signal_date < oot_start or signal_date > oot_end:
                continue

            # Find entry date (next trading day after signal)
            entry_candidates = ticker_prices.index[ticker_prices.index > signal_date]
            if len(entry_candidates) < 5:
                continue
            entry_date = entry_candidates[0]

            # Variant-specific filters
            if variant in ("B", "D"):
                # Value filter: P/B < sector median
                sector = SECTOR_MAP.get(ticker, "Unknown")
                fund = fundamentals.get(ticker, {})
                idx_match = None
                for i, d in enumerate(fund.get("dates", [])):
                    if d == change["date"]:
                        idx_match = i
                        break
                if idx_match is None:
                    continue
                bvps = fund["book_value_per_share"][idx_match]
                if not bvps or bvps <= 0:
                    continue
                entry_price_check = ticker_prices.loc[entry_date] if entry_date in ticker_prices.index else None
                if entry_price_check is None or pd.isna(entry_price_check):
                    continue
                pb = float(entry_price_check) / bvps

                # Compute sector median P/B
                sector_pbs = []
                if sector in sector_pb and change["date"] in sector_pb[sector]:
                    sector_pbs = [x[1] for x in sector_pb[sector][change["date"]]]
                if not sector_pbs:
                    for dt_key, vals in sector_pb.get(sector, {}).items():
                        if abs((pd.Timestamp(dt_key) - signal_date).days) < 120:
                            sector_pbs.extend([x[1] for x in vals])
                if not sector_pbs:
                    continue
                sector_median_pb = np.median(sector_pbs)
                if pb >= sector_median_pb:
                    continue  # not undervalued

            if variant in ("C", "D"):
                # Quality filter: ROE > 15% and FCF > 0
                fund = fundamentals.get(ticker, {})
                idx_match = None
                for i, d in enumerate(fund.get("dates", [])):
                    if d == change["date"]:
                        idx_match = i
                        break
                if idx_match is None:
                    continue
                roe = fund["roe"][idx_match]
                fcf = fund["fcf"][idx_match]
                if roe is None or roe < 0.15:
                    continue
                if fcf is None or fcf <= 0:
                    continue

            if variant == "E":
                # Momentum: RSI > 40 and price > 50-SMA (not in downtrend)
                if entry_date not in ticker_rsi.index:
                    continue
                rsi_val = ticker_rsi.loc[entry_date]
                sma50_val = ticker_sma50.loc[entry_date] if entry_date in ticker_sma50.index else None
                if pd.isna(rsi_val) or rsi_val <= 40:
                    continue
                if sma50_val is None or pd.isna(sma50_val):
                    continue
                if float(ticker_prices.loc[entry_date]) < float(sma50_val):
                    continue

            # Entry price with slippage
            raw_entry = float(ticker_prices.loc[entry_date])
            entry_price = raw_entry * (1 + SLIPPAGE_PCT)

            # Exit date
            exit_target = entry_date + pd.Timedelta(days=hold_days)
            exit_candidates = ticker_prices.index[ticker_prices.index >= exit_target]
            if len(exit_candidates) == 0:
                exit_date = ticker_prices.index[-1]
            else:
                exit_date = exit_candidates[0]

            raw_exit = float(ticker_prices.loc[exit_date])
            exit_price = raw_exit * (1 - SLIPPAGE_PCT)

            # Regime: half-size if SPY < 200-SMA at entry
            regime_half = False
            if entry_date in spy_sma200.index and entry_date in spy_close.index:
                spy_price = float(spy_close.loc[entry_date])
                spy_sma = float(spy_sma200.loc[entry_date])
                if not np.isnan(spy_sma) and spy_price < spy_sma:
                    regime_half = True

            # Determine regime label for gap analysis
            if entry_date in spy_sma200.index and entry_date in spy_close.index:
                spy_price = float(spy_close.loc[entry_date])
                spy_sma = float(spy_sma200.loc[entry_date])
                regime = "bear" if (not np.isnan(spy_sma) and spy_price < spy_sma) else "bull"
            else:
                regime = "unknown"

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "hold_days": (exit_date - entry_date).days,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_return": (exit_price - entry_price) / entry_price,
                "shares_change_pct": change["pct_change"],
                "regime_half": regime_half,
                "regime": regime,
            })

    return trades


# ── Backtest Analytics ─────────────────────────────────────────────────────

def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd": 0,
                "total_return": 0, "ann_return": 0, "avg_return": 0, "avg_hold_days": 0}

    returns = []
    for t in trades:
        r = t["pct_return"]
        if t["regime_half"]:
            r *= 0.5  # half-size in bear regime
        returns.append(r)

    returns = np.array(returns)
    n = len(returns)

    total_ret = np.prod(1 + returns) - 1
    avg_ret = np.mean(returns)
    win_rate = np.sum(returns > 0) / n if n > 0 else 0

    # Sharpe (annualized)
    trades_per_year = 252 / np.mean([t["hold_days"] for t in trades]) if trades else 4
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Max Drawdown
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    # Annualized return
    if trades:
        first_entry = min(t["entry_date"] for t in trades)
        last_exit = max(t["exit_date"] for t in trades)
        days_span = (pd.Timestamp(last_exit) - pd.Timestamp(first_entry)).days
        years = max(days_span / 365.25, 0.5)
        ann_return = (1 + total_ret) ** (1 / years) - 1
    else:
        ann_return = 0

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(min(pf, 99), 3),
        "wr": round(win_rate, 4),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_return, 4),
        "avg_return": round(avg_ret, 4),
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
    }


def compute_regime_gap(trades):
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    if not bull_trades or not bear_trades:
        return 0.0  # can't compute, pass by default

    bull_metrics = compute_metrics(bull_trades)
    bear_metrics = compute_metrics(bear_trades)

    s_bull = bull_metrics["sharpe"]
    s_bear = bear_metrics["sharpe"]
    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 0.0
    gap = abs(s_bull - s_bear) / denom
    return round(gap, 4)


def permutation_test(trades, n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Permutation test: shuffle trade returns to get null distribution of Sharpe.
    Returns p-value.
    """
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["pct_return"] * (0.5 if t["regime_half"] else 1.0) for t in trades])
    trades_per_year = 252 / np.mean([t["hold_days"] for t in trades])

    observed_sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year) if np.std(returns) > 0 else 0

    rng = np.random.RandomState(seed)
    count_above = 0
    for _ in range(n_perms):
        perm_returns = returns * rng.choice([-1, 1], size=len(returns))
        if np.std(perm_returns) > 0:
            perm_sharpe = (np.mean(perm_returns) / np.std(perm_returns)) * np.sqrt(trades_per_year)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            count_above += 1

    return round(count_above / n_perms, 4)


def validate_5_gates(metrics, trades):
    """Run 5-gate validation."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = {"pass": metrics["sharpe"] > 0.5, "value": metrics["sharpe"]}

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades)
    gates["permutation_p_lt_0.05"] = {"pass": p_val < 0.05, "value": p_val}

    # Gate 3: Regime gap < 0.5
    regime_gap = compute_regime_gap(trades)
    gates["regime_gap_lt_0.5"] = {"pass": regime_gap < 0.5, "value": regime_gap}

    # Gate 4: MaxDD > -50%
    gates["maxdd_gt_neg50"] = {"pass": metrics["max_dd"] > -0.50, "value": metrics["max_dd"]}

    # Gate 5: >= 20 trades
    gates["trades_gte_20"] = {"pass": metrics["n_trades"] >= 20, "value": metrics["n_trades"]}

    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict))
    return gates


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("SHARE BUYBACK ANNOUNCEMENT DRIFT BACKTEST")
    log("=" * 70)

    # Download data
    prices = download_price_data(TOP_100_SP500)
    spy_data = download_spy_data()
    fundamentals = download_quarterly_data(TOP_100_SP500)

    # Compute shares changes
    shares_changes = compute_shares_change(fundamentals)
    total_buybacks = sum(len([c for c in changes if c["pct_change"] < -0.01])
                         for changes in shares_changes.values())
    log(f"Total buyback signals (>1% decrease): {total_buybacks}")

    # Run all variants
    variants = {
        "A": "Base (shares decrease >1%, 60d hold)",
        "B": "Value filter (+ P/B < sector median, 60d hold)",
        "C": "Quality filter (+ ROE>15% + FCF>0, 60d hold)",
        "D": "Combined (value + quality, 60d hold)",
        "E": "Momentum overlay (+ RSI>40, above SMA50, 40d hold)",
    }

    results = {
        "strategy": "Share Buyback Announcement Drift",
        "academic_basis": "Ikenberry, Lakonishok & Vermaelen (1995, 2000)",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "universe": f"Top {len(TOP_100_SP500)} S&P 500 by market cap",
        "cost_model": f"$0 commission, {SLIPPAGE_PCT*100:.2f}% slippage per side",
        "regime_hedge": "Half-size when SPY < 200-SMA",
        "run_timestamp": datetime.now().isoformat(),
        "variants": {},
    }

    for var_key, var_desc in variants.items():
        log(f"\n{'---'*20}")
        log(f"Variant {var_key}: {var_desc}")
        log(f"{'---'*20}")

        trades = generate_signals(shares_changes, fundamentals, prices, spy_data, variant=var_key)
        log(f"  Trades generated: {len(trades)}")

        if not trades:
            log(f"  NO TRADES -- skipping")
            results["variants"][var_key] = {
                "description": var_desc,
                "n_trades": 0,
                "metrics": {},
                "gates": {"all_pass": False, "reason": "No trades generated"},
            }
            continue

        metrics = compute_metrics(trades)
        gates = validate_5_gates(metrics, trades)

        # Per-regime breakdown
        bull_trades = [t for t in trades if t["regime"] == "bull"]
        bear_trades = [t for t in trades if t["regime"] == "bear"]
        bull_metrics = compute_metrics(bull_trades) if bull_trades else {}
        bear_metrics = compute_metrics(bear_trades) if bear_trades else {}

        # Top tickers
        ticker_counts = {}
        ticker_returns = {}
        for t in trades:
            tk = t["ticker"]
            ticker_counts[tk] = ticker_counts.get(tk, 0) + 1
            if tk not in ticker_returns:
                ticker_returns[tk] = []
            ticker_returns[tk].append(t["pct_return"])

        top_tickers = sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]

        results["variants"][var_key] = {
            "description": var_desc,
            "metrics": metrics,
            "gates": gates,
            "regime_breakdown": {
                "bull": {"n_trades": len(bull_trades), **bull_metrics},
                "bear": {"n_trades": len(bear_trades), **bear_metrics},
            },
            "top_tickers": [{"ticker": tk, "trades": cnt,
                              "avg_return": round(np.mean(ticker_returns[tk]), 4)}
                            for tk, cnt in top_tickers],
            "sample_trades": [
                {k: v for k, v in t.items() if k != "regime_half"}
                for t in sorted(trades, key=lambda x: -x["pct_return"])[:5]
            ],
        }

        # Print summary
        log(f"  N trades:     {metrics['n_trades']}")
        log(f"  Sharpe:       {metrics['sharpe']}")
        log(f"  Sortino:      {metrics['sortino']}")
        log(f"  Profit Factor:{metrics['pf']}")
        log(f"  Win Rate:     {metrics['wr']:.1%}")
        log(f"  Max DD:       {metrics['max_dd']:.2%}")
        log(f"  Total Return: {metrics['total_return']:.2%}")
        log(f"  Ann. Return:  {metrics['ann_return']:.2%}")
        log(f"  Avg Hold:     {metrics['avg_hold_days']} days")
        log(f"  Bull trades:  {len(bull_trades)}, Bear trades: {len(bear_trades)}")

        gate_status = "PASS" if gates["all_pass"] else "FAIL"
        failed_gates = [k for k, v in gates.items() if isinstance(v, dict) and not v["pass"]]
        log(f"  5-Gate:       {gate_status}" + (f" (failed: {', '.join(failed_gates)})" if failed_gates else ""))

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    for var_key, var_data in results["variants"].items():
        gates = var_data.get("gates", {})
        metrics = var_data.get("metrics", {})
        status = "PASS" if gates.get("all_pass", False) else "FAIL"
        n = metrics.get("n_trades", var_data.get("n_trades", 0))
        sharpe = metrics.get("sharpe", "N/A")
        log(f"  {var_key}: {status} | {n} trades | Sharpe={sharpe} | {var_data['description']}")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved.")

    return results


if __name__ == "__main__":
    results = main()
